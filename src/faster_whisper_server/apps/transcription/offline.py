from collections.abc import Generator
from pathlib import Path
from re import I

import gradio as gr

from faster_whisper_server.apps.transcription.client import HttpTranscriberClient
from faster_whisper_server.apps.transcription.compare import add_compare_ui
from faster_whisper_server.apps.transcription.i18n import I18nText
from faster_whisper_server.config import Config


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
                    yield previous_transcription
            else:
                yield self.http_client.post(model, language, temperature, file)

    @classmethod
    def create_gradio_interface(cls, config: Config, model_dropdown, language, temperature_slider, stream_checkbox):
        offline_transcription = cls(config.host, config.port)
        gr.Markdown("""
### 离线转码
- 功能: 上传音频文件,系统一次性将音频发送到ASR后台处理,模拟离线转码
- 用途: 测试ASR后台的离线转码性能
- 用法: 上传音频文件后,点击Start按扭
""")
        audio = gr.Audio(type="filepath")
        btn = gr.Button(I18nText("开始", "Start"))
        text = gr.Textbox(
            label=I18nText(
                "识别结果(红色为当前确认结果,绿色为未确认结果)",
                "Transcription (Red for current confirmed, Green for unconfirmed)",
            ),
            interactive=False,
            show_inline_category=False,
            show_legend=False,
            combine_adjacent=True,
            color_map={"confirmed": "red", "unconfirmed": "green", "info": "gray"},
        )
        btn.click(
            offline_transcription.on_click,
            inputs=[audio, model_dropdown, language, temperature_slider, stream_checkbox],
            outputs=[text],
        )
        add_compare_ui(text)
