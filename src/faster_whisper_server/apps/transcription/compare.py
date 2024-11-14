from difflib import Differ

import gradio as gr

from faster_whisper_server.apps.transcription.i18n import I18nText


def add_compare_ui(text, is_heighlighted=False):
    ground_truth = gr.Textbox(label=I18nText("真实结果", "Ground Truth"))
    compare_btn = gr.Button(I18nText("比较", "Compare"))
    diff = gr.HighlightedText(combine_adjacent=True, label=I18nText("差异(+/-)", "Diff(+/-)"), show_legend=True)

    def diff_texts(prediction_text, reference_text):
        if is_heighlighted:
            prediction_text = prediction_text[0]["token"]
        d = Differ()
        return [
            (token[2:], token[0] if token[0] != " " else None) for token in d.compare(reference_text, prediction_text)
        ]

    compare_btn.click(
        diff_texts,
        inputs=[text, ground_truth],
        outputs=[diff],
    )
