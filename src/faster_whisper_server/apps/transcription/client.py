import logging
from typing import Generator
from urllib import response
from urllib.parse import urlencode

import httpx
import msgpack
import websockets
from httpx_sse import connect_sse
from pydub import AudioSegment

logger = logging.getLogger(__name__)

TRANSCRIPTION_ENDPOINT = "/v1/audio/transcriptions"
TIMEOUT_SECONDS = 180
TIMEOUT = httpx.Timeout(timeout=TIMEOUT_SECONDS)


class WebSocketTranscriberClient:
    def __init__(self, port: int, host: str, endpoint=TRANSCRIPTION_ENDPOINT):
        self.port = port
        self.host = host
        self.base_url = f"ws://{host}:{port}{TRANSCRIPTION_ENDPOINT}"

        self.ws = None

    def encode_query(self, model, language, temperature):
        queries = {
            "response_format": "text",
            "model": model,
            "temperature": temperature,
        }
        if language != "auto":
            queries["language"] = language
        return f"{self.base_url}?{urlencode(queries)}"

    async def connect(self, model: str, language: str, temperature: float):
        self.ws = await websockets.connect(self.encode_query(model, language, temperature))
        return self

    async def disconnect(self):
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def send_audio_chunk(self, audio_chunk: AudioSegment):
        await self.send_raw_data(audio_chunk.raw_data)

    async def send_raw_data(self, raw_data: bytes):
        await self.ws.send(msgpack.packb({"data": raw_data}))

    async def send_stop_command(self):
        logger.info("Sending stop command")
        await self.ws.send(msgpack.packb({"stop": True}))

    async def receive_response(self):
        return msgpack.unpackb(await self.ws.recv())

    async def receive_transcription(self) -> tuple[str, str]:
        response = await self.receive_response()
        confirmed = response["confirmed"]
        unconfirmed = response["unconfirmed"]
        return confirmed, unconfirmed


class HttpTranscriberClient:
    def __init__(self, port: int, host: str, endpoint=TRANSCRIPTION_ENDPOINT, timeout=TIMEOUT):
        self.port = port
        self.host = host
        self.endpoint = endpoint
        self.base_url = f"http://{host}:{port}"
        self.http_client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def post(self, model: str, language: str, temperature: float, file, response_format="text") -> tuple[str, str]:
        data = {
            "response_format": response_format,
            "model": model,
            "temperature": temperature,
        }
        if language != "auto":
            data["language"] = language
        response = self.http_client.post(
            self.endpoint,
            files={"file": file},
            data=data,
        )
        response.raise_for_status()
        if response_format == "text":
            return response.text, language
        elif response_format == "verbose_json":
            from faster_whisper_server.api_models import CreateTranscriptionResponseVerboseJson, segments_to_text

            response = CreateTranscriptionResponseVerboseJson.model_validate_json(response.text)
            return segments_to_text(response.segments), response.language
        elif response_format == "json":
            return response.json(), language
        else:
            raise ValueError(f"Unknown response format: {response_format}")

    def sse_post(self, model: str, language: str, temperature: float, file) -> Generator[str, None, None]:
        with connect_sse(
            client=self.http_client,
            method="POST",
            url=self.endpoint,
            files={"file": file},
            data={
                "response_format": "text",
                "model": model,
                "temperature": temperature,
                "stream": "true",
            },
        ) as event_source:
            for event in event_source.iter_sse():
                yield event.data
