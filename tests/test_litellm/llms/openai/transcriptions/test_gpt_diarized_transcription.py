import io
from unittest.mock import MagicMock

import httpx

from litellm.llms.openai.transcriptions.gpt_transformation import (
    OpenAIGPTAudioTranscriptionConfig,
)
from litellm.utils import get_optional_params_transcription


def test_public_transcription_params_preserve_diarization_fields():
    optional = get_optional_params_transcription(
        model="gpt-4o-transcribe-diarize",
        custom_llm_provider="openai",
        response_format="diarized_json",
        chunking_strategy="auto",
        known_speaker_names=["Alice"],
        known_speaker_references=["data:audio/wav;base64,AAAA"],
    )

    expected = {
        "response_format": "diarized_json",
        "chunking_strategy": "auto",
        "known_speaker_names": ["Alice"],
        "known_speaker_references": ["data:audio/wav;base64,AAAA"],
    }
    assert {key: optional[key] for key in expected} == expected
    assert optional.get("extra_body") in ({}, None)


def test_gpt_diarized_request_preserves_chunking_and_speaker_fields():
    config = OpenAIGPTAudioTranscriptionConfig()
    audio = io.BytesIO(b"audio")
    audio.name = "meeting.webm"

    request = config.transform_audio_transcription_request(
        model="gpt-4o-transcribe-diarize",
        audio_file=audio,
        optional_params={
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
            "known_speaker_names": ["Alice"],
            "known_speaker_references": ["data:audio/wav;base64,AAAA"],
        },
        litellm_params={},
    )

    assert request.data["response_format"] == "diarized_json"
    assert request.data["chunking_strategy"] == "auto"
    assert request.data["known_speaker_names"] == ["Alice"]
    assert request.data["known_speaker_references"] == ["data:audio/wav;base64,AAAA"]


def test_gpt_diarized_response_serializes_segments_and_usage():
    config = OpenAIGPTAudioTranscriptionConfig()
    raw = MagicMock(spec=httpx.Response)
    raw.json.return_value = {
        "text": "Hello there",
        "language": "en",
        "duration": 2.5,
        "segments": [{"speaker": "A", "start": 0.0, "end": 2.5, "text": "Hello there"}],
        "usage": {"type": "duration", "seconds": 3},
    }

    response = config.transform_audio_transcription_response(raw)
    serialized = response.model_dump()

    assert serialized["segments"][0]["speaker"] == "A"
    assert serialized["language"] == "en"
    assert serialized["duration"] == 2.5
    assert serialized["usage"]["seconds"] == 3
