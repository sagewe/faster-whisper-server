from collections.abc import Generator
from pathlib import Path

import gradio as gr

from faster_whisper_server.apps.transcription.client import HttpTranscriberClient
from faster_whisper_server.apps.transcription.compare import add_compare_ui
from faster_whisper_server.apps.transcription.i18n import I18nText
from faster_whisper_server.apps.transcription.style import TRANSCRIPTION_HEIGHLIGHT_STYLE
from faster_whisper_server.config import Config

DESCRIPTION = I18nText(
    """
### 离线转码
- 功能: 上传音频文件,系统一次性将音频发送到ASR后台处理,模拟离线转码
- 用途: 测试ASR后台的离线转码性能, 上传音频文件后, 点击开始按扭
""",
    """
### Offline Transcription
- Function: Upload an audio file, the system sends the audio to the ASR backend for processing at once, simulating offline transcoding
- Usage: Test the performance of offline transcoding of the ASR backend, upload an audio file, click the Start button
""",
)


class OfflineTranscription:
    def __init__(self, host, port):
        self.http_client = HttpTranscriberClient(port, host)

    def on_click(
        self, file_path: str, model: str, language: str, temperature: float, stream: bool
    ) -> Generator[str, None, None]:
        with Path(file_path).open("rb") as file:
            if stream:
                previous_transcription = ""
                for text in self.http_client.sse_post(model, language, temperature, file):
                    previous_transcription += text
                    yield [(previous_transcription, "confirmed")]
            else:
                text, _ = self.http_client.post(model, language, temperature, file)
                yield [(text, "confirmed")]

    @classmethod
    def create_gradio_interface(cls, config: Config, model_dropdown, language, temperature_slider, stream_checkbox):
        offline_transcription = cls(config.host, config.port)
        gr.Markdown(DESCRIPTION)
        audio = gr.Audio(type="filepath")
        btn = gr.Button(I18nText("开始", "Start"))
        text = gr.HighlightedText(**TRANSCRIPTION_HEIGHLIGHT_STYLE)
        btn.click(
            offline_transcription.on_click,
            inputs=[audio, model_dropdown, language, temperature_slider, stream_checkbox],
            outputs=[text],
        )
        # add_compare_ui(text)
