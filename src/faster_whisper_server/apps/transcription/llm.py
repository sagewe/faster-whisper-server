import io
import logging
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import gradio as gr
import librosa
import numpy as np
from gtts import gTTS
from openai import OpenAI, chat
from pydub import AudioSegment
from pathlib import Path
from regex import F
import soundfile

from .qa import RAGLLM, punctuation

from faster_whisper_server.utils.vad import VadOptions, collect_chunks, get_speech_timestamps
from faster_whisper_server.apps.transcription.client import HttpTranscriberClient
from faster_whisper_server.config import Config


logger = logging.getLogger(__name__)

# recording parameters
IN_CHANNELS = 1
IN_RATE = 24000
IN_CHUNK = 1024
IN_SAMPLE_WIDTH = 2
VAD_STRIDE = 0.5

# playing parameters
OUT_CHANNELS = 1
OUT_RATE = 24000
OUT_SAMPLE_WIDTH = 2
OUT_CHUNK = 5760


OUT_CHUNK = 20 * 4096
OUT_RATE = 24000
OUT_CHANNELS = 1


@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_detected: bool = False
    started_talking: bool = False
    stopped: bool = False
    conversation_visual: list = field(default_factory=list)
    qa: Optional[RAGLLM] = None
    asr_input: Optional[str] = None
    asr_result: Optional[str] = None
    tts_language: Optional[str] = None


desc2lang = {
    "粤语": "yue",
    "香港粤语": "yue",
    "普通话": "zh",
    "English": "en",
    # 阿拉伯语
    "العربية": "ar",
    # 越南语
    "Tiếng Việt": "vi",
    # 泰语
    "ไทย": "th",
}


class AudioChatBot:
    def __init__(self, host, port):
        self.http_client = HttpTranscriberClient(port, host)

    def create_app_state(self, conversation_visual=None, stopped=False, qa=None):
        kwargs = {"stopped": stopped}
        if conversation_visual is not None:
            kwargs["conversation_visual"] = conversation_visual
        if qa is not None:
            kwargs["qa"] = qa
        return AppState(**kwargs)

    def run_vad(self, ori_audio, sr):
        _st = time.time()
        try:
            audio = ori_audio
            audio = audio.astype(np.float32) / 32768.0
            sampling_rate = 16000
            if sr != sampling_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

            vad_parameters = {"threshold": 0.6}
            vad_parameters = VadOptions(**vad_parameters)
            speech_chunks = get_speech_timestamps(audio, vad_parameters)
            audio = collect_chunks(audio, speech_chunks)
            duration_after_vad = audio.shape[0] / sampling_rate

            if sr != sampling_rate:
                # resample to original sampling rate
                vad_audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sr)
            else:
                vad_audio = audio
            vad_audio = np.round(vad_audio * 32768.0).astype(np.int16)
            vad_audio_bytes = vad_audio.tobytes()

            return duration_after_vad, vad_audio_bytes, round(time.time() - _st, 4)
        except Exception as e:
            msg = f"[asr vad error] audio_len: {len(ori_audio)/(sr*2):.3f} s, trace: {traceback.format_exc()}"
            print(msg)
            return -1, ori_audio, round(time.time() - _st, 4)

    def warm_up(self):
        frames = b"\x00\x00" * 1024 * 2  # 1024 frames of 2 bytes each
        frames = np.frombuffer(frames, dtype=np.int16)
        dur, frames, tcost = self.run_vad(frames, 16000)
        print(f"warm up done, time_cost: {tcost:.3f} s")

    def determine_pause(self, audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
        """Take in the stream, determine if a pause happened"""

        temp_audio = audio

        dur_vad, _, time_vad = self.run_vad(temp_audio, sampling_rate)
        duration = len(audio) / sampling_rate

        if dur_vad > 0.5 and not state.started_talking:
            print("started talking")
            state.started_talking = True
            return False

        print(f"duration_after_vad: {dur_vad:.3f} s, time_vad: {time_vad:.3f} s")

        return (duration - dur_vad) > 1

    def on_stream(self, audio: tuple, state: AppState):
        logger.info("on_stream, state: %s", state)
        if state.stream is None:
            state.stream = audio[1]
            state.sampling_rate = audio[0]
        else:
            state.stream = np.concatenate((state.stream, audio[1]))

        # pause_detected = self.determine_pause(state.stream, state.sampling_rate, state)
        # state.pause_detected = pause_detected

        # if state.pause_detected and state.started_talking:
        #     return gr.Audio(recording=False), state
        return state

    def on_stop_recording(
        self,
        state: AppState,
        vectorstore_path: str,
        vectorstore: str,
        chat_model: str,
        embedding_model: str,
        dashscope_api_key: str,
    ):
        # if not state.started_talking:
        #     state = self.create_app_state(qa=state.qa)
        #     return None, state, state.conversation_visual
        logger.info("on_stop_recording, state: %s", state)

        if state.qa is None:
            state.qa = RAGLLM(
                vectorstore_path=vectorstore_path,
                collection_name=vectorstore,
                chat_model=chat_model,
                embedding_model=embedding_model,
                api_key=dashscope_api_key,
                streaming=True,
            )
        return state

    def on_dump_user_audio(self, state: AppState, add_chat_audio: bool):
        if state.asr_input is not None:
            return state, state.conversation_visual
        # save recorded audio to file
        logger.info("on_dump_user_audio, state: %s", state)
        start_time = time.time()
        logger.info("Start saving user audio to wav file")
        audio_buffer = io.BytesIO()
        segment = AudioSegment(
            state.stream.tobytes(),
            frame_rate=state.sampling_rate,
            sample_width=state.stream.dtype.itemsize,
            channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
        )
        segment.export(audio_buffer, format="wav")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_buffer.getvalue())
        state.asr_input = f.name

        if add_chat_audio:
            state.conversation_visual.append(
                {"role": "user", "content": {"path": state.asr_input, "mime_type": "audio/wav"}}
            )
        logger.info(f"User audio saved to {f.name}, time cost: {time.time() - start_time:.3f} s")
        return state, state.conversation_visual

    def on_asr(self, state: AppState, asr_model: str, asr_language: str, asr_temperature: float):
        if state.asr_result is not None:
            return state, state.conversation_visual

        # ASR
        logger.info("on_asr, state: %s", state)
        start_time = time.time()
        logger.info("Start ASR")
        with Path(state.asr_input).open("rb") as f:
            asr_result, asr_language = self.http_client.post(
                asr_model, asr_language, asr_temperature, f, response_format="verbose_json"
            )

        user_asr_message = {"role": "user", "content": asr_result}
        state.conversation_visual.append(user_asr_message)
        state.asr_result = asr_result
        state.tts_language = asr_language
        logger.info(f"ASR done, time cost: {time.time() - start_time:.3f} s")
        return state, state.conversation_visual

    def on_answer(self, state: AppState, speed_rate: float, add_chat_audio: bool):
        logger.info("on_answer, state: %s", state)
        # LLM
        start_time = time.time()
        logger.info("Start AI response")

        def get_tts(text, language, speed_rate):
            tts = gTTS(text, lang=language)
            with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
                tts.write_to_fp(f)
                librosa_audio, sr = librosa.load(f.name, sr=None)
                librosa_audio = librosa.effects.time_stretch(librosa_audio, rate=speed_rate)
                return sr, librosa_audio

        cache = ""
        full_tts = np.array([])
        state.conversation_visual.append({"role": "assistant", "content": ""})
        for chunk in state.qa.stream(state.asr_result, state.tts_language):
            state.conversation_visual[-1]["content"] += chunk
            cache += chunk
            if cache and cache[-1] in punctuation:
                try:
                    sr, librosa_audio = get_tts(cache, state.tts_language, speed_rate)
                    full_tts = np.concatenate((full_tts, librosa_audio))
                    yield state, (sr, np.int16(librosa_audio * 32768.0)), state.conversation_visual
                except Exception as e:
                    print(e)
                cache = ""
            else:
                yield state, None, state.conversation_visual
        if cache:
            try:
                sr, librosa_audio = get_tts(cache, state.tts_language, speed_rate)
                full_tts = np.concatenate((full_tts, librosa_audio))
                yield state, (sr, np.int16(librosa_audio * 32768.0)), state.conversation_visual
            except Exception as e:
                print(e)

        if add_chat_audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                soundfile.SoundFile(f.name, mode="w", samplerate=sr, channels=1, subtype="PCM_16").write(full_tts)
            state.conversation_visual.append(
                {"role": "assistant", "content": {"path": f.name, "mime_type": "audio/wav"}}
            )
        logger.info(f"AI response done, time cost: {time.time() - start_time:.3f} s")

        yield (
            self.create_app_state(conversation_visual=state.conversation_visual, qa=state.qa),
            None,
            state.conversation_visual,
        )

    def on_start_recording_user(self, state: AppState):
        if not state.stopped:
            return gr.Audio(recording=True)

    @classmethod
    def create_gradio_interface(cls, config: Config, model_dropdown, language_dropdown, temperature_slider):
        audio_chatbot = cls(host=config.host, port=config.port)
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="Conversation", type="messages", height=600)
            with gr.Column(scale=1):
                with gr.Accordion(label="语音设置"):
                    speed_rate_slider = gr.Slider(minimum=0.5, maximum=3.0, step=0.1, label="语速", value=1.5)
                    input_audio = gr.Audio(label="语音输入", sources="microphone", type="numpy")
                    output_audio = gr.Audio(label="语音输出", type="numpy", autoplay=True, streaming=True)
                with gr.Accordion(label="对话设置", open=False):
                    chat_model = gr.Dropdown(
                        choices=["qwen-max"],
                        label="对话模型",
                        value="qwen-max",
                    )
                    vectorstore_path = gr.Textbox(
                        value=config.vectorstore_path,
                        label="Vectorstore路径",
                        placeholder="请输入Vectorstore路径",
                        visible=False,
                    )
                    vectorstore_dropdown = gr.Dropdown(
                        choices=["银行与保险知识库"],
                        label="知识库",
                        value="银行与保险知识库",
                    )
                    embedding_model = gr.Dropdown(
                        choices=["text-embedding-v1"],
                        label="Embedding模型",
                        value="text-embedding-v1",
                    )
                    api_key_textbox = gr.Textbox(value=config.dashscope_api_key, label="API Key", type="password")
                    add_chat_audio = gr.Checkbox(label="添加对话音频", value=False)
                cancel = gr.Button("重置对话", variant="stop")
        state = gr.State(value=audio_chatbot.create_app_state())

        stream = input_audio.stream(
            audio_chatbot.on_stream,
            [input_audio, state],
            [state],
            stream_every=0.50,
            time_limit=30,
        )
        respond = (
            input_audio.stop_recording(
                audio_chatbot.on_stop_recording,
                [
                    state,
                    vectorstore_path,
                    vectorstore_dropdown,
                    chat_model,
                    embedding_model,
                    api_key_textbox,
                ],
                [state],
            )
            .success(
                audio_chatbot.on_dump_user_audio,
                [state, add_chat_audio],
                [state, chatbot],
            )
            .success(
                audio_chatbot.on_asr,
                [state, model_dropdown, language_dropdown, temperature_slider],
                [state, chatbot],
            )
            .success(
                audio_chatbot.on_answer,
                [state, speed_rate_slider, add_chat_audio],
                [state, output_audio, chatbot],
            )
        )
        respond.then(
            lambda: gr.Audio(interactive=True),
            [],
            [input_audio],
        )
        cancel.click(
            lambda: (audio_chatbot.create_app_state(stopped=True), gr.Audio(recording=False)),
            None,
            [state, input_audio],
            cancels=[respond],
        )
