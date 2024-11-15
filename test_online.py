import os
import datasets
from gradio_client import Client, handle_file
import tqdm
import soundfile as sf
import evaluate

base_dir = os.path.dirname(__file__)

def online_transcriptions(
    dataset: datasets.Dataset,
    url: str,
    save_path: str,
    model: str = "deepdml/faster-whisper-large-v3-turbo-ct2",
    language: str = "ar",
):
    transcriptions = {}
    latency = []
    wer = evaluate.load(f"{os.path.join(base_dir, 'evaluate-metric', 'wer')}")

    def process_row(i, row):
        audio = row["audio"]
        sf.write(f"tmp/{i}.wav", audio["array"], audio["sampling_rate"], format="WAV")
        client = Client(url, verbose=False)
        result = client.predict(api_name="/update_model_dropdown")
        result = client.predict(model=model, api_name="/update_language_dropdown")
        result = client.predict(
            audio_file=handle_file(f"tmp/{i}.wav"),
            model=model,
            language=language,
            temperature=0.0,
            api_name="/stream_audio_file",
        )
        prediction = result[0]["token"]
        latency.append(float(result[1]["token"].lstrip("\nLatency: ").rstrip("ms")))
        reference = row["sentence"] if "sentence" in row else row["transcription"]
        wer.add(prediction=prediction, reference=reference)
        transcriptions[i] = prediction
        time.sleep(3)

    os.makedirs("tmp/", exist_ok=True)
    for i in tqdm.tqdm(range(len(dataset))):
        for _ in range(3):
            try:
                import time

                process_row(i, dataset[i])
                break
            except Exception as e:
                print(e)
    
    print(f"the WER of {model} is {wer.compute()}")

    def fn_dataset_with_transcriptions(row, i):
        row["online_transcription"] = transcriptions[i]
        return row

    dataset_with_transcriptions = dataset.map(fn_dataset_with_transcriptions, with_indices=True)
    dataset_with_transcriptions.save_to_disk(save_path)
    import numpy as np
    print(f"latency: mean={np.mean(latency)}, 50%={np.percentile(latency, 50)}, 90%={np.percentile(latency, 90)}")


URL = "http://127.0.0.1:8000/"


def run_data(dataset_name, model, sample_num=-1):
    print(f"process {dataset_name} with model {model}, sample_num={sample_num}")
    if dataset_name == "common_voice_15_0":
        dataset = datasets.load_from_disk(f"{os.path.join(base_dir, 'dataset/mozilla-foundation/common_voice_15_0')}")
    elif dataset_name == "fleurs":
        dataset = datasets.load_from_disk(f"{os.path.join(base_dir, 'dataset/google/fleurs')}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if sample_num > 0:
        dataset = dataset.select(range(sample_num))
    save_path = f"outputs/{dataset_name}/{model}"

    online_transcriptions(
        dataset=dataset,
        model=model,
        url=URL,
        language="ar",
        save_path=save_path,
    )


for model in [
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "Systran/faster-whisper-large-v3",
]:
    run_data("fleurs", model, -1)
    run_data("common_voice_15_0", model, 500)
