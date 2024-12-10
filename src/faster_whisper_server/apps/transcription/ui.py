import logging

import gradio as gr
import httpx
from openai import OpenAI

from faster_whisper_server.apps.transcription.i18n import I18nText
from faster_whisper_server.apps.transcription.live import LiveTranscription
from faster_whisper_server.apps.transcription.offline import OfflineTranscription
from faster_whisper_server.apps.transcription.simulated_live import SimulatedLiveTranscription
from faster_whisper_server.apps.transcription.llm import AudioChatBot
from faster_whisper_server.config import Config


logger = logging.getLogger(__name__)

MODEL_LOAD_ENDPOINT = "/api/ps"
TIMEOUT_SECONDS = 180
TIMEOUT = httpx.Timeout(timeout=TIMEOUT_SECONDS)


def create_gradio_demo(config: Config) -> gr.Blocks:
    base_url = f"http://{config.host}:{config.port}"
    openai_client = OpenAI(base_url=f"{base_url}/v1", api_key="cant-be-empty")
    http_client = httpx.Client(base_url=base_url, timeout=TIMEOUT)

    def update_model_dropdown() -> gr.Dropdown:
        model_data = openai_client.models.list().data
        mapping = {"cwj": 0, "deepdml": 1, "Systran": 2}
        models = {model.id: model for model in model_data}
        model_names = list(models.keys())
        model_names.sort(key=lambda x: mapping.get(x.split("/")[0], 3))
        dropdown = gr.Dropdown(
            value=model_names[0],
            choices=model_names,
            label="Model",
        )
        return dropdown, models

    def update_language_dropdown(model, models) -> gr.Dropdown:
        model_data = models[model]
        languages = model_data.language
        languages.append("auto")
        mapping = {"yue": 0, "zh": 1, "ar": 2, "en": 3, "auto": 4}
        languages.sort(key=lambda x: mapping.get(x, 5))
        dropdown = gr.Dropdown(
            value=languages[0],
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
# 客服对话演示系统
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
                with gr.Tab(I18nText("问答机器人", "Chatbot")):
                    chatbot = AudioChatBot.create_gradio_interface(
                        config, model_dropdown, language_dropdown, temperature_slider
                    )
                # with gr.Tab(I18nText("模拟实时转码", "Simulated Live Transcription")):
                #     simulated_live_transcription = SimulatedLiveTranscription.create_gradio_interface(
                #         config, model_dropdown, language_dropdown, temperature_slider
                #     )
                # with gr.Tab(I18nText("离线转码", "Offline Transcription")):
                #     offline_transcription = OfflineTranscription.create_gradio_interface(
                #         config, model_dropdown, language_dropdown, temperature_slider, stream_checkbox
                #     )

                # with gr.Tab(I18nText("实时转码", "Real-time Transcription")):
                #     live_transcription = LiveTranscription.create_gradio_interface(
                #         config, model_dropdown, language_dropdown, temperature_slider
                #     )

    return iface
