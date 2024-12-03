# import os

# import gradio as gr
# from langchain.chat_models import ChatOpenAI
# from langchain.schema import AIMessage, HumanMessage
# from openai import OpenAI

# llm = ChatOpenAI(temperature=1.0, model='gpt-4o-mini')

# def predict(message, history):
#     history_langchain_format = []
#     for msg in history:
#         if msg['role'] == "user":
#             history_langchain_format.append(HumanMessage(content=msg['content']))
#         elif msg['role'] == "assistant":
#             history_langchain_format.append(AIMessage(content=msg['content']))
#     history_langchain_format.append(HumanMessage(content=message))
#     gpt_response = llm(history_langchain_format)
#     return gpt_response.content

# gr.ChatInterface(predict, type="messages").launch()


# if __name__ == "__main__":
#     gr.ChatInterface(predict, type="messages").launch()


from io import BytesIO
import gradio as gr
import time
import gtts

from RealtimeTTS import TextToAudioStream, GTTSEngine, AzureEngine, ElevenlabsEngine
import pydub
from regex import B

engine = GTTSEngine()
stream = TextToAudioStream(engine, muted=True)


def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})

    return history, gr.MultimodalTextbox(value=None, interactive=False)


def bot(history: list):
    FRAME_RATE = 22050
    response = "That's cool!"
    history.append({"role": "assistant", "content": ""})

    stream.feed(response)
    chunks = []

    def on_audio_chunk(chunk):
        chunks.append(chunk)

    stream.play(on_audio_chunk=on_audio_chunk, muted=True)
    history[-1]["content"] = response
    import numpy as np

    audio_buffer = BytesIO(b"".join(chunks))
    audio = pydub.AudioSegment.from_raw(audio_buffer, sample_width=2, frame_rate=FRAME_RATE, channels=2).set_channels(1)

    return history, (FRAME_RATE, np.array(audio.get_array_of_samples()))


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
    )
    chat_response_audio = gr.Audio(label="response", autoplay=True, type="numpy", visible=False)
    chat_request_audio = gr.Audio(label="request", sources=["upload", "microphone"], visible=True)
    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, [chatbot, chat_response_audio], api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

demo.launch()
