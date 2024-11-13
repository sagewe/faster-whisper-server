from collections.abc import Generator
from pathlib import Path

import gradio as gr

from faster_whisper_server.apps.transcription.client import HttpTranscriberClient
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
        btn = gr.Button("Start")
        text = gr.Textbox(label="Transcription", interactive=False)
        btn.click(
            offline_transcription.on_click,
            inputs=[audio, model_dropdown, language, temperature_slider, stream_checkbox],
            outputs=[text],
        )
        with gr.Accordion(open=False, label="Compare"):
            from difflib import Differ

            ground_truth = gr.Textbox(label="Ground Truth")
            compare_btn = gr.Button("Compare")
            diff = gr.HighlightedText(combine_adjacent=True, label="Diff")

            def diff_texts(text1, text2):
                d = Differ()
                return [(token[2:], token[0] if token[0] != " " else None) for token in d.compare(text1, text2)]

            compare_btn.click(
                diff_texts,
                inputs=[text, ground_truth],
                outputs=[diff],
            )
