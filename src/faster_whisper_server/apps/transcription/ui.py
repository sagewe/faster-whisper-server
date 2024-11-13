import asyncio
import logging
import traceback
from collections import deque

import gradio as gr
import httpx
import websockets
from openai import OpenAI
from pydub import AudioSegment

from faster_whisper_server.apps.transcription.const import SAMPLES_RATE
from faster_whisper_server.apps.transcription.offline import OfflineTranscription
from faster_whisper_server.apps.transcription.simulated_live import SimulatedLiveTranscription
from faster_whisper_server.config import Config

from .client import WebSocketTranscriberClient

logger = logging.getLogger(__name__)

MODEL_LOAD_ENDPOINT = "/api/ps"
TIMEOUT_SECONDS = 180
TIMEOUT = httpx.Timeout(timeout=TIMEOUT_SECONDS)
TRANSCRIPTION_UPDATE_INTERVAL = 0.1


class SessionAudioStreamer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.ws_client = WebSocketTranscriberClient(port, host)

        self.ws_established = asyncio.Event()
        self.current_confirmed = ""
        self.current_unconfirmed = ""
        self.is_closing = False
        self.close_timeout = 3.0

        self.send_task = None
        self.receive_task = None
        self.buffer = deque()

    async def pre_close(self):
        self.is_closing = True
        try:
            if self.buffer:
                combined_data = b"".join(self.buffer)
                await self.ws_client.send_raw_data(combined_data)
                self.buffer.clear()
            await self.ws_client.send_stop_command()
        except Exception as e:
            logger.error(f"Error sending final buffer: {e}")

    async def close(self):
        try:
            if self.send_task:
                self.send_task.cancel()
            if self.receive_task:
                self.receive_task.cancel()
            await self.ws_client.disconnect()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

    async def _send_buffer(self):
        logger.info("Starting to send buffer")
        await self.ws_established.wait()
        logger.info("WebSocket connection established for session")
        while not self.is_closing:  # 检查关闭标志
            if self.buffer:
                combined_data = b"".join(self.buffer)
                self.buffer.clear()
                try:
                    logger.info(f"sending {len(combined_data)} bytes to WebSocket")
                    await self.ws_client.send_raw_data(combined_data)
                    logger.info("Sent buffer")
                except websockets.ConnectionClosed:
                    break
            else:
                await asyncio.sleep(0.1)
        logger.info("Finished sending buffer")

    async def _receive_responses(self):
        await self.ws_established.wait()
        while True:
            try:
                self.current_confirmed, self.current_unconfirmed = await self.ws_client.receive_transcription()
            except websockets.ConnectionClosed:
                logger.info("WebSocket connection closed")
                break

    async def buffer_audio_chunk(self, audio_chunk, model, language, temperature):
        if not self.ws_established.is_set():
            await self.initialize_ws_connection(model, language, temperature)
            self.send_task = asyncio.create_task(self._send_buffer())
            self.receive_task = asyncio.create_task(self._receive_responses())

        sr, data = audio_chunk
        audio_chunk = (
            AudioSegment(data.tobytes(), frame_rate=sr, sample_width=data.dtype.itemsize, channels=1)
            .set_frame_rate(SAMPLES_RATE)
            .raw_data
        )
        self.buffer.append(audio_chunk)

    async def initialize_ws_connection(self, model: str, language: str, temperature: float):
        await self.ws_client.connect(model=model, language=language, temperature=temperature)
        self.ws_established.set()
        logger.info("WebSocket connection established for session")


def create_gradio_demo(config: Config) -> gr.Blocks:
    base_url = f"http://{config.host}:{config.port}"
    openai_client = OpenAI(base_url=f"{base_url}/v1", api_key="cant-be-empty")
    http_client = httpx.Client(base_url=base_url, timeout=TIMEOUT)

    def update_model_dropdown() -> gr.Dropdown:
        model_data = openai_client.models.list().data
        models = {model.id: model for model in model_data}
        model_names = list(models.keys())
        dropdown = gr.Dropdown(
            value="deepdml/faster-whisper-large-v3-turbo-ct2",
            choices=model_names,
            label="Model",
        )
        return dropdown, models

    def update_language_dropdown(model, models) -> gr.Dropdown:
        model_data = models[model]
        languages = model_data.language
        value = "auto"
        if len(languages) == 1:
            value = languages[0]
            languages = [value]
        else:
            mapping = {"en": 0, "zh": 1, "ar": 2}
            languages.sort(key=lambda x: mapping.get(x, 3))
            languages = ["auto"] + languages
        dropdown = gr.Dropdown(
            value=value,
            choices=languages,
            label="Language",
        )
        return dropdown

    def fn_preload_models(model):
        response = http_client.post(f"{MODEL_LOAD_ENDPOINT}/{model}")
        if response.is_success:
            return "Models preloaded successfully"
        elif response.status_code == 409:
            return "Model already loaded"
        else:
            return "Failed to preload models"

    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        models = gr.State({})
        gr.Markdown("""
# ASR 演示系统
""")
        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    label="Model",
                )
                language_dropdown = gr.Dropdown(
                    label="Language",
                )
                temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.0)
                stream_checkbox = gr.Checkbox(label="Stream", value=True)
                preload_models = gr.Button("Preload models")
                preload_models_reponse = gr.Textbox(show_label=False)
                iface.load(update_model_dropdown, inputs=[], outputs=[model_dropdown, models])
                model_dropdown.change(
                    update_language_dropdown, inputs=[model_dropdown, models], outputs=[language_dropdown]
                )
                preload_models.click(fn_preload_models, inputs=[model_dropdown], outputs=[preload_models_reponse])
            with gr.Column(scale=3):
                with gr.Tab("Live Transcript Simulation"):
                    simulated_live_transcription = SimulatedLiveTranscription.create_gradio_interface(
                        config, model_dropdown, language_dropdown, temperature_slider
                    )
                with gr.Tab("Offline Transcript"):
                    offline_transcription = OfflineTranscription.create_gradio_interface(
                        config, model_dropdown, language_dropdown, temperature_slider, stream_checkbox
                    )

                with gr.Tab("Live Transcript"):
                    gr.Markdown(
                        """### 实时转码
- 功能: 读取麦克风信号,实时输出转码
- 用途: 演示实时转码功能
- 用法: 点击麦克风图标开始接收音频,转码结果会实时显示在Transcription文本框中
- 注意: 系统当前确认的结果用红色显示,未确认的结果用绿色显示
"""
                    )
                    streamer_state = gr.State(lambda: SessionAudioStreamer(host=config.host, port=config.port))
                    audio_live = gr.Audio(sources=["microphone"], type="numpy", streaming=True)
                    timer = gr.Timer(value=TRANSCRIPTION_UPDATE_INTERVAL, active=False)
                    text_live_output = gr.HighlightedText(
                        label="Transcription",
                        interactive=False,
                        show_inline_category=False,
                        show_legend=False,
                        combine_adjacent=True,
                        color_map={"confirmed": "red", "unconfirmed": "green", "info": "gray"},
                    )

                    async def process_audio_stream(
                        audio_chunk, state: SessionAudioStreamer, model, language, temperature
                    ):
                        try:
                            if audio_chunk is not None:
                                await state.buffer_audio_chunk(audio_chunk, model, language, temperature)
                                return gr.update(active=True)
                        except Exception:
                            return traceback.format_exc()

                    async def on_stop_recording(state: SessionAudioStreamer):
                        logger.info("Stopping session")
                        await state.pre_close()
                        logger.info("Session closed")

                    async def on_tick(state: SessionAudioStreamer):
                        if state.is_closing:
                            state.close_timeout -= TRANSCRIPTION_UPDATE_INTERVAL

                        if state.close_timeout <= 0:
                            logger.info("stop timer")
                            await state.close()
                            return [
                                (state.current_confirmed, "confirmed"),
                                (state.current_unconfirmed, "unconfirmed"),
                            ], gr.update(active=False)

                        return [
                            (state.current_confirmed, "confirmed"),
                            (state.current_unconfirmed, "unconfirmed"),
                        ], gr.skip()

                    audio_live.stream(
                        fn=process_audio_stream,
                        inputs=[audio_live, streamer_state, model_dropdown, language_dropdown, temperature_slider],
                        outputs=timer,
                        show_progress=False,
                        api_name=None,
                        time_limit=30,
                        stream_every=0.1,
                    )
                    audio_live.stop_recording(fn=on_stop_recording, inputs=[streamer_state])
                    timer.tick(fn=on_tick, inputs=[streamer_state], outputs=[text_live_output, timer])

                # TODO: HighlightedText does not support RTL, show we contribute to Gradio?
                # def fn_update_rtl(language_dropdown):
                #     rtl = language_dropdown == "ar"
                #     return gr.update(rtl=rtl)

                # language_dropdown.change(
                #     fn_update_rtl,
                #     inputs=[language_dropdown],
                #     outputs=[text_transcription_output],
                # )
    return iface
