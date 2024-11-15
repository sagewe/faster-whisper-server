import asyncio
import logging
import time

import gradio as gr
import websockets
from pydub import AudioSegment

from faster_whisper_server.apps.transcription.client import WebSocketTranscriberClient
from faster_whisper_server.apps.transcription.compare import add_compare_ui
from faster_whisper_server.apps.transcription.const import CHUNK_TIME, SAMPLES_RATE, TRANSCRIPTION_COLOR_MAPPING
from faster_whisper_server.apps.transcription.i18n import I18nText
from faster_whisper_server.apps.transcription.style import TRANSCRIPTION_HEIGHLIGHT_STYLE
from faster_whisper_server.config import Config

logger = logging.getLogger(__name__)

DESCRIPTION = I18nText(
    """
### 模拟实时转码
- 功能: 上传音频文件,系统按照音频实际的播放速度连续发送音频到ASR后台,模拟实时ASR
- 用途: 模拟实时转码,测试ASR后台的实时性能, 上传音频文件后, 点击播放按扭
- 注意: 系统当前确认的结果用红色显示,未确认的结果用绿色显示
""",
    """
### Simulated Live Transcription
- Function: Upload an audio file, the system will send audio to the ASR backend continuously according to the actual playback speed of the audio, simulating real-time ASR
- Usage: Simulate real-time transcoding, test the real-time performance of the ASR backend, upload an audio file, click the play button
- Note: The current confirmed result is displayed in red, and the unconfirmed result is displayed in green
""",
)


class SimulatedLiveTranscription:
    def __init__(self, host, port, chunk_time_ms: int = CHUNK_TIME, sample_rate=SAMPLES_RATE, channel=1):
        self.ws_client = WebSocketTranscriberClient(port, host)
        self.chunk_time_ms = chunk_time_ms
        self.sample_rate = sample_rate
        self.channel = channel

    async def stream_audio(self, audio_file_path):
        logger.info(f"Streaming audio file: {audio_file_path}")
        audio = AudioSegment.from_file(audio_file_path).set_frame_rate(self.sample_rate).set_channels(self.channel)
        start = time.perf_counter()
        time_per_chunk_s = self.chunk_time_ms / 1000.0
        num_chunks = len(audio) // self.chunk_time_ms + 1

        for i, audio_start in enumerate(range(0, len(audio), self.chunk_time_ms)):
            audio_chunk = audio[audio_start : audio_start + self.chunk_time_ms]
            await self.ws_client.send_audio_chunk(audio_chunk)
            if i < num_chunks - 1:
                await asyncio.sleep((i + 1) * time_per_chunk_s - (time.perf_counter() - start))

        last_chunk_send_time = time.perf_counter()
        await self.ws_client.send_stop_command()
        return last_chunk_send_time

    async def stream_audio_file(self, audio_file, model, language, temperature):
        confirmed = ""
        try:
            await self.ws_client.connect(model=model, language=language, temperature=temperature)
            stream_task = asyncio.create_task(self.stream_audio(audio_file))
            try:
                while True:
                    confirmed, unconfirmed = await self.ws_client.receive_transcription()
                    last_chunk_response_time = time.perf_counter()
                    yield [(confirmed, "confirmed"), (unconfirmed, "unconfirmed")]
            except websockets.ConnectionClosed:
                pass
            last_packet_request_time = await stream_task
        finally:
            await self.ws_client.disconnect()

        yield [
            (confirmed, "confirmed"),
            (f"\nLatency: {(last_chunk_response_time - last_packet_request_time) * 1000:.1f} ms", "info"),
        ]

    @classmethod
    def create_gradio_interface(cls, config: Config, model_dropdown, language_dropdown, temperature_slider):
        simulated_live_transcription = cls(config.host, config.port)
        gr.Markdown(DESCRIPTION)
        audio = gr.Audio(label=I18nText("音频", "Audio"), type="filepath")
        text = gr.HighlightedText(**TRANSCRIPTION_HEIGHLIGHT_STYLE)
        audio.play(
            simulated_live_transcription.stream_audio_file,
            inputs=[audio, model_dropdown, language_dropdown, temperature_slider],
            outputs=[text],
        )
        # add_compare_ui(text, is_heighlighted=True)
