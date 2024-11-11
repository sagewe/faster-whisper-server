import asyncio
from calendar import c
from io import BytesIO
from re import A, S
import re
import time
from collections import deque
from pathlib import Path
from typing import Generator
from urllib.parse import urlencode

import gradio as gr
import httpx
import numpy as np
import websockets
from httpx_sse import connect_sse
from openai import OpenAI
from pydub import AudioSegment
import traceback

from faster_whisper_server.config import Config
import logging

logger = logging.getLogger(__name__)

# Audio parameters
SAMPLES_RATE = 16000
CHUNK_TIME = 100  # Chunk size in ms
TRANSCRIPTION_ENDPOINT = "/v1/audio/transcriptions"
MODEL_LOAD_ENDPOINT = "/api/ps"
TRANSLATION_ENDPOINT = "/v1/audio/translations"
TIMEOUT_SECONDS = 180
TIMEOUT = httpx.Timeout(timeout=TIMEOUT_SECONDS)

css = """
#rtl-textbox1{
  text-align: right;
}
#rtl-textbox2{
  text-align: right;
}
#rtl-textbox3{
  text-align: right;
}
"""


def create_gradio_demo(config: Config) -> gr.Blocks:
    base_url = f"http://{config.host}:{config.port}"
    openai_client = OpenAI(base_url=f"{base_url}/v1", api_key="cant-be-empty")
    http_client = httpx.Client(base_url=base_url, timeout=TIMEOUT)
    WEBSOCKET_URL_BASE = f"ws://{config.host}:{config.port}/v1/audio/transcriptions"

    async def receive_responses(ws):
        """Receive responses from the WebSocket server asynchronously."""
        try:
            while True:
                response = await ws.recv()
                yield response
        except websockets.ConnectionClosed:
            return

    async def stream_audio(ws, audio_file_path):
        """Stream audio from the specified file to the WebSocket server."""
        audio = AudioSegment.from_file(audio_file_path).set_frame_rate(SAMPLES_RATE).set_channels(1)
        start = time.perf_counter()
        expect_time_per_chunk = CHUNK_TIME / 1000
        for i, audio_start in enumerate(range(0, len(audio), CHUNK_TIME)):
            audio_chunk = audio[audio_start : audio_start + CHUNK_TIME]
            # buffer = BytesIO()
            # audio_chunk.export(buffer, format="raw")
            await ws.send(audio_chunk.raw_data)
            time_to_sleep = (i + 1) * expect_time_per_chunk - (time.perf_counter() - start)
            await asyncio.sleep(time_to_sleep)

    async def websocket_stream(audio_file_path, base, queries):
        url = f"{base}?{urlencode(queries)}"
        async with websockets.connect(url) as ws:
            stream_task = asyncio.create_task(stream_audio(ws, audio_file_path))

            # Yield responses live as they arrive
            async for response in receive_responses(ws):
                yield response

    async def stream_audio_file(audio_file, model, language, temperature):
        queries = {
            "response_format": "text",
            "model": model,
            "temperature": temperature,
        }
        if language != "auto":
            queries["language"] = language
        async for response in websocket_stream(audio_file, WEBSOCKET_URL_BASE, queries):
            yield response  # Yield each WebSocket response to Gradio

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

    async def fn_stream(audio_stream, model, language, temperature):
        async for stream in audio_stream:
            print(stream)

    ## Audio Transcription
    def handler(file_path: str, model: str, temperature: float, stream: bool) -> Generator[str, None, None]:
        endpoint = TRANSCRIPTION_ENDPOINT

        if stream:
            previous_transcription = ""
            for transcription in streaming_audio_task(file_path, endpoint, temperature, model):
                previous_transcription += transcription
                yield previous_transcription
        else:
            yield audio_task(file_path, endpoint, temperature, model)

    def audio_task(file_path: str, endpoint: str, temperature: float, model: str) -> str:
        with Path(file_path).open("rb") as file:
            response = http_client.post(
                endpoint,
                files={"file": file},
                data={
                    "model": model,
                    "response_format": "text",
                    "temperature": temperature,
                },
            )

        response.raise_for_status()
        return response.text

    def streaming_audio_task(
        file_path: str, endpoint: str, temperature: float, model: str
    ) -> Generator[str, None, None]:
        with Path(file_path).open("rb") as file:
            kwargs = {
                "files": {"file": file},
                "data": {
                    "response_format": "text",
                    "temperature": temperature,
                    "model": model,
                    "stream": True,
                },
            }
            with connect_sse(http_client, "POST", endpoint, **kwargs) as event_source:
                for event in event_source.iter_sse():
                    yield event.data

    class SessionAudioStreamer:
        def __init__(self):
            self.buffer = deque()
            self.ws_connection = None
            self.ws_established = asyncio.Event()
            self.send_task = None
            self.receive_task = None
            self.current_response = ""  # 存储最新的响应
            self.is_closing = False  # 添加关闭标志

        async def close(self):
            """优雅关闭连接"""
            self.is_closing = True

            # 1. 等待缓冲区发送完成
            if self.buffer:
                try:
                    if self.ws_connection and self.buffer:
                        combined_data = b"".join(self.buffer)
                        await self.ws_connection.send(combined_data)
                        self.buffer.clear()
                except Exception as e:
                    logger.error(f"Error sending final buffer: {e}")

            # 2. 等待5秒接收最后的响应
            try:
                await asyncio.sleep(5)
            finally:
                # 3. 取消任务并关闭连接
                if self.send_task:
                    self.send_task.cancel()
                if self.receive_task:
                    self.receive_task.cancel()
                if self.ws_connection:
                    await self.ws_connection.close()

            return self.current_response  # 返回最终的响应

        async def _send_buffer(self):
            logger.info("Starting to send buffer")
            await self.ws_established.wait()
            logger.info("WebSocket connection established for session")
            while not self.is_closing:  # 检查关闭标志
                if self.ws_connection and self.buffer:
                    combined_data = b"".join(self.buffer)
                    self.buffer.clear()
                    try:
                        logger.info(f"sending {len(combined_data)} bytes to WebSocket")
                        await self.ws_connection.send(combined_data)
                        logger.info("Sent buffer")
                    except websockets.ConnectionClosed:
                        break
                else:
                    await asyncio.sleep(0.1)
            logger.info("Finished sending buffer")

        async def _receive_responses(self):
            """负责接收WebSocket服务器的响应"""
            await self.ws_established.wait()
            while True:
                if self.ws_connection:
                    try:
                        response = await self.ws_connection.recv()
                        logger.info(f"received response: {response}")
                        self.current_response = response
                    except websockets.ConnectionClosed:
                        logger.info("WebSocket connection closed")
                        break
                else:
                    await asyncio.sleep(0.1)

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

        async def initialize_ws_connection(self, model, language, temperature):
            """Initialize the WebSocket connection based on session parameters."""

            queries = {
                "response_format": "text",
                "model": model,
                "temperature": temperature,
            }
            if language != "auto":
                queries["language"] = language

            url = f"{WEBSOCKET_URL_BASE}?{urlencode(queries)}"
            self.ws_connection = await websockets.connect(url)
            self.ws_established.set()
            logger.info("WebSocket connection established for session")

    with gr.Blocks(theme=gr.themes.Soft(), css=css) as iface:
        models = gr.State({})
        gr.Markdown("""
# ASR 演示系统
本系统支持三种演示模式：
1. 实时模拟：上传音频文件，系统实际音频的播放速度连续发送音频到ASR后台，模拟实时ASR
2. 音频转码：上传音频文件，系统一次性将音频发送到ASR后台处理，模拟离线转码
3. 实时演示：直接读取麦克风信号，进行实时转码
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
            with gr.Column():
                with gr.Tab("Audio Live Simulation"):
                    audio_file_simulation = gr.Audio(label="Audio", sources=["upload"], type="filepath")
                    text_simulation_output = gr.Textbox(label="Transcription", interactive=False, rtl=True)
                    audio_file_simulation.play(
                        stream_audio_file,
                        inputs=[audio_file_simulation, model_dropdown, language_dropdown, temperature_slider],
                        outputs=[text_simulation_output],
                    )

                with gr.Tab("Audio Transcript"):
                    audio_file_transcript = gr.Audio(type="filepath")
                    btn = gr.Button("Start")
                    text_transcription_output = gr.Textbox(label="Transcription", interactive=False, rtl=True)
                    btn.click(
                        handler,
                        inputs=[audio_file_transcript, model_dropdown, temperature_slider, stream_checkbox],
                        outputs=[text_transcription_output],
                    )

                with gr.Tab("Live"):
                    streamer_state = gr.State(lambda: SessionAudioStreamer())
                    audio_live = gr.Audio(sources=["microphone"], type="numpy", streaming=True)
                    text_live_output = gr.Textbox(label="Transcription", interactive=False, rtl=True)

                    async def process_audio_stream(
                        audio_chunk, state: SessionAudioStreamer, model, language, temperature
                    ):
                        try:
                            if audio_chunk is None:
                                return ""

                            await state.buffer_audio_chunk(audio_chunk, model, language, temperature)
                            return state.current_response
                        except Exception as e:
                            return traceback.format_exc()

                    async def on_stop(state: SessionAudioStreamer):
                        try:
                            if state:
                                final_response = await state.close()
                                return final_response  # 返回最终响应
                        except Exception as e:
                            print(f"Error in on_stop: {e}")
                            return str(e)

                    audio_live.stream(
                        fn=process_audio_stream,
                        inputs=[audio_live, streamer_state, model_dropdown, language_dropdown, temperature_slider],
                        outputs=text_live_output,
                        show_progress=False,
                        api_name=None,
                        time_limit=1,
                    )

                    audio_live.stop_recording(fn=on_stop, inputs=[streamer_state], outputs=text_live_output)

                def fn_update_rtl(
                    language_dropdown, text_simulation_output, text_transcription_output, text_live_output
                ):
                    rtl = language_dropdown == "ar"
                    return gr.update(rtl=rtl), gr.update(rtl=rtl), gr.update(rtl=rtl)

                language_dropdown.change(
                    fn_update_rtl,
                    inputs=[language_dropdown],
                    outputs=[text_simulation_output, text_transcription_output, text_live_output],
                )
    return iface
