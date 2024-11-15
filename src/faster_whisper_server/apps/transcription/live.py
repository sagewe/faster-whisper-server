import asyncio
import logging
import traceback
from collections import deque

import gradio as gr
import websockets
from pydub import AudioSegment

from faster_whisper_server.apps.transcription.client import WebSocketTranscriberClient
from faster_whisper_server.apps.transcription.compare import add_compare_ui
from faster_whisper_server.apps.transcription.const import SAMPLES_RATE, TRANSCRIPTION_UPDATE_INTERVAL
from faster_whisper_server.apps.transcription.i18n import I18nText
from faster_whisper_server.apps.transcription.style import TRANSCRIPTION_HEIGHLIGHT_STYLE
from faster_whisper_server.config import Config

logger = logging.getLogger(__name__)


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


DESCRIPTION = I18nText(
    """### 实时转码
- 功能: 读取麦克风信号,实时输出转码
- 用途: 演示实时转码功能, 点击麦克风图标开始接收音频
- 注意: 系统当前确认的结果用绿色显示,未确认的结果用红色显示
""",
    """### Live Transcription
- Function: Read microphone signal, output transcoding in real time
- Usage: Demonstrate real-time transcoding function, click the microphone icon to start receiving audio
- Note: The current confirmed result is displayed in red, and the unconfirmed result is displayed in green
""",
)


class LiveTranscription:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    @classmethod
    def create_gradio_interface(cls, config: Config, model_dropdown, language_dropdown, temperature_slider):
        live_transcription = cls(config.host, config.port)

        gr.Markdown(DESCRIPTION)
        streamer_state = gr.State(
            lambda: SessionAudioStreamer(host=live_transcription.host, port=live_transcription.port)
        )
        audio_live = gr.Audio(sources=["microphone"], type="numpy", streaming=True)
        timer = gr.Timer(value=TRANSCRIPTION_UPDATE_INTERVAL, active=False)
        text_live_output = gr.HighlightedText(**TRANSCRIPTION_HEIGHLIGHT_STYLE)

        async def process_audio_stream(audio_chunk, state: SessionAudioStreamer, model, language, temperature):
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
        # add_compare_ui(text_live_output, is_heighlighted=True)
