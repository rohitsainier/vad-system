"""
TTS Engine - Text-to-Speech via IndicF5 Voice Cloning API
"""
import numpy as np
import base64
import io
import wave
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TTSResult:
    """Result from TTS synthesis"""
    audio: np.ndarray
    sample_rate: int
    duration_ms: float
    engine_used: str
    used_seed: Optional[int] = None
    reference_voice_info: Optional[dict] = None


@dataclass
class ReferenceVoice:
    """Reference voice metadata from IndicF5"""
    key: str
    author: str
    content: str
    file: str
    sample_rate: int
    model: str


class IndicF5Client:
    """
    HTTP client for the IndicF5 TTS voice-cloning API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 120,
        default_reference_voice_key: str = "",
        default_sample_rate: int = 24000,       # ✅ FIX: was 16000
        default_output_format: str = "wav",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_reference_voice_key = default_reference_voice_key
        self.default_sample_rate = default_sample_rate
        self.default_output_format = default_output_format

        self._session = None
        self._reference_voices_cache: Optional[Dict[str, ReferenceVoice]] = None

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"Content-Type": "application/json"})
        return self._session

    def _post_json(self, path: str, payload: dict) -> dict:
        session = self._get_session()
        url = f"{self.base_url}{path}"
        logger.debug("indicf5_request", url=url, payload_keys=list(payload.keys()))
        resp = session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get_json(self, path: str) -> dict:
        session = self._get_session()
        url = f"{self.base_url}{path}"
        resp = session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> dict:
        session = self._get_session()
        url = f"{self.base_url}{path}"
        resp = session.delete(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _b64_wav_to_numpy(audio_b64: str) -> Tuple[np.ndarray, int]:
        """Decode base64 WAV to numpy, reading the ACTUAL sample rate from
        the WAV header so we never guess wrong."""
        wav_bytes = base64.b64decode(audio_b64)
        with io.BytesIO(wav_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

        if sampwidth == 2:
            dtype, max_val = np.int16, 32768.0
        elif sampwidth == 4:
            dtype, max_val = np.int32, 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32) / max_val
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        logger.debug(
            "wav_decoded",
            sample_rate=sr,
            samples=len(audio),
            duration_s=round(len(audio) / sr, 3),
        )
        return audio, sr

    # ------------------------------------------------------------------ #
    #  Health
    # ------------------------------------------------------------------ #

    def health_check(self) -> bool:
        try:
            self._get_json("/api/health")
            return True
        except Exception as exc:
            logger.debug("indicf5_health_fail", error=str(exc))
            return False

    # ------------------------------------------------------------------ #
    #  Reference voices
    # ------------------------------------------------------------------ #

    def list_reference_voices(self, force_refresh: bool = False) -> Dict[str, ReferenceVoice]:
        if self._reference_voices_cache is not None and not force_refresh:
            return self._reference_voices_cache

        data = self._get_json("/api/referenceVoices")
        voices: Dict[str, ReferenceVoice] = {}
        for key, info in data.get("reference_voices", {}).items():
            voices[key] = ReferenceVoice(
                key=key,
                author=info.get("author", ""),
                content=info.get("content", ""),
                file=info.get("file", ""),
                sample_rate=info.get("sample_rate", 24000),   # ✅ FIX
                model=info.get("model", "F5TTS"),
            )
        self._reference_voices_cache = voices
        return voices

    def upload_reference_voice(
        self,
        filepath: str,
        name: str,
        author: str,
        content: str = "",
        model: str = "F5TTS",
    ) -> dict:
        session = self._get_session()
        url = f"{self.base_url}/api/referenceVoices/upload"
        with open(filepath, "rb") as f:
            files = {"file": (Path(filepath).name, f)}
            form = {"name": name, "author": author, "content": content, "model": model}
            resp = session.post(url, files=files, data=form,
                                timeout=self.timeout, headers={})
        resp.raise_for_status()
        self._reference_voices_cache = None
        return resp.json()

    def delete_reference_voice(self, voice_key: str) -> dict:
        result = self._delete(f"/api/referenceVoices/{voice_key}")
        self._reference_voices_cache = None
        return result

    def get_reference_voice_audio_url(self, voice_key: str) -> str:
        return f"{self.base_url}/api/referenceVoices/{voice_key}/audio"

    # ------------------------------------------------------------------ #
    #  Single TTS
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        text: str,
        reference_voice_key: Optional[str] = None,
        output_format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        normalize: bool = True,
        seed: int = -1,
        save_to_file: bool = False,
    ) -> dict:
        payload = {
            "text": text,
            "reference_voice_key": reference_voice_key or self.default_reference_voice_key,
            "output_format": output_format or self.default_output_format,
            "sample_rate": sample_rate or self.default_sample_rate,
            "normalize": normalize,
            "seed": seed,
            "save_to_file": save_to_file,
        }
        return self._post_json("/api/tts", payload)

    def synthesize_to_numpy(
        self,
        text: str,
        reference_voice_key: Optional[str] = None,
        output_format: str = "wav",
        sample_rate: Optional[int] = None,
        normalize: bool = True,
        seed: int = -1,
    ) -> Tuple[np.ndarray, int, dict]:
        resp = self.synthesize(
            text=text,
            reference_voice_key=reference_voice_key,
            output_format=output_format,
            sample_rate=sample_rate,
            normalize=normalize,
            seed=seed,
            save_to_file=False,
        )
        if not resp.get("success"):
            raise RuntimeError(
                f"IndicF5 synthesis failed: {resp.get('message', 'unknown error')}"
            )
        audio, sr = self._b64_wav_to_numpy(resp["audio_base64"])
        return audio, sr, resp

    # ------------------------------------------------------------------ #
    #  Batch TTS
    # ------------------------------------------------------------------ #

    def synthesize_batch(
        self,
        requests_list: List[Dict[str, Any]],
        return_as_zip: bool = False,
    ) -> dict:
        payload = {
            "requests": requests_list,
            "return_as_zip": return_as_zip,
        }
        return self._post_json("/api/tts/batch", payload)

    # ------------------------------------------------------------------ #
    #  Prompt-tagged TTS
    # ------------------------------------------------------------------ #

    def synthesize_prompt_tagged(
        self,
        text: str,
        base_reference_voice_key: Optional[str] = None,
        output_format: str = "wav",
        sample_rate: Optional[int] = None,
        normalize: bool = True,
        max_chunk_chars: int = 300,
        pause_duration: int = 200,
    ) -> dict:
        payload = {
            "text": text,
            "base_reference_voice_key": base_reference_voice_key
            or self.default_reference_voice_key,
            "output_format": output_format,
            "sample_rate": sample_rate or self.default_sample_rate,
            "normalize": normalize,
            "max_chunk_chars": max_chunk_chars,
            "pause_duration": pause_duration,
        }
        return self._post_json("/api/tts/prompt-tagged", payload)

    # ------------------------------------------------------------------ #
    #  Podcast
    # ------------------------------------------------------------------ #

    def generate_podcast(
        self,
        title: str,
        speakers: List[Dict[str, str]],
        segments: List[Dict[str, Any]],
        pause_duration: int = 500,
        output_format: str = "wav",
        sample_rate: Optional[int] = None,
        normalize: bool = True,
        seed: int = -1,
    ) -> dict:
        payload = {
            "title": title,
            "speakers": speakers,
            "segments": segments,
            "pause_duration": pause_duration,
            "output_format": output_format,
            "sample_rate": sample_rate or self.default_sample_rate,
            "normalize": normalize,
            "seed": seed,
        }
        return self._post_json("/api/podcast/generate", payload)

    # ------------------------------------------------------------------ #
    #  File management
    # ------------------------------------------------------------------ #

    def list_files(self) -> dict:
        return self._get_json("/api/files")

    def download_file_url(self, filename: str) -> str:
        return f"{self.base_url}/api/files/{filename}"

    def delete_file(self, filename: str) -> dict:
        return self._delete(f"/api/files/{filename}")

    def clear_all_files(self) -> dict:
        return self._delete("/api/files")

    # ------------------------------------------------------------------ #
    #  System
    # ------------------------------------------------------------------ #

    def system_monitor(self) -> dict:
        return self._get_json("/api/system/monitor")

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    def close(self):
        if self._session is not None:
            self._session.close()
            self._session = None


class TTSEngine:
    """
    Text-to-Speech engine powered by the IndicF5 voice-cloning API.
    """

    def __init__(
        self,
        indicf5_base_url: str = "http://localhost:8000",
        indicf5_reference_voice_key: str = "",
        indicf5_sample_rate: int = 24000,       # ✅ FIX: was 16000
        indicf5_output_format: str = "wav",
        indicf5_timeout: int = 120,
        indicf5_seed: int = -1,
        indicf5_normalize: bool = True,
    ):
        self.indicf5_client = IndicF5Client(
            base_url=indicf5_base_url,
            timeout=indicf5_timeout,
            default_reference_voice_key=indicf5_reference_voice_key,
            default_sample_rate=indicf5_sample_rate,
            default_output_format=indicf5_output_format,
        )
        self._indicf5_reference_voice_key = indicf5_reference_voice_key
        self._indicf5_sample_rate = indicf5_sample_rate
        self._indicf5_seed = indicf5_seed
        self._indicf5_normalize = indicf5_normalize
        self._is_initialized = False

    # ================================================================== #
    #  Initialization
    # ================================================================== #

    def initialize(self):
        """Verify the IndicF5 server is reachable."""
        if self._is_initialized:
            return

        if not self.indicf5_client.health_check():
            raise RuntimeError(
                f"IndicF5 server not reachable at {self.indicf5_client.base_url}. "
                "Make sure the server is running."
            )

        logger.info(
            "indicf5_ready",
            url=self.indicf5_client.base_url,
            voice=self._indicf5_reference_voice_key or "(auto)",
            sample_rate=self._indicf5_sample_rate,
        )
        self._is_initialized = True

    # ================================================================== #
    #  Core synthesis
    # ================================================================== #

    def synthesize(
        self,
        text: str,
        reference_voice_key: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> TTSResult:
        """
        Synthesize text to speech using IndicF5 voice cloning.

        Args:
            text:                Text to convert.
            reference_voice_key: Override the default reference voice.
            seed:                Override the default seed (-1 = random).

        Returns:
            TTSResult with float32 audio numpy array.
        """
        if not self._is_initialized:
            self.initialize()

        if not text.strip():
            sr = self._indicf5_sample_rate
            silence_samples = int(sr * 0.1)  # 100ms of silence
            return TTSResult(
                audio=np.zeros(silence_samples, dtype=np.float32),
                sample_rate=sr,
                duration_ms=100.0,
                engine_used="F5TTS",
            )

        voice_key = (
            reference_voice_key
            or self._indicf5_reference_voice_key
            or self.indicf5_client.default_reference_voice_key
        )

        # Auto-pick first available voice if none configured
        if not voice_key:
            voices = self.indicf5_client.list_reference_voices()
            if not voices:
                raise RuntimeError(
                    "No reference voices on the IndicF5 server. "
                    "Upload one with upload_reference_voice()."
                )
            voice_key = next(iter(voices))
            logger.info("indicf5_auto_voice", voice_key=voice_key)

        audio, actual_sr, meta = self.indicf5_client.synthesize_to_numpy(
            text=text,
            reference_voice_key=voice_key,
            output_format="wav",
            sample_rate=self._indicf5_sample_rate,
            normalize=self._indicf5_normalize,
            seed=seed if seed is not None else self._indicf5_seed,
        )

        # ✅ Warn if the server returned a different rate than requested
        if actual_sr != self._indicf5_sample_rate:
            logger.warning(
                "indicf5_sample_rate_mismatch",
                requested=self._indicf5_sample_rate,
                actual=actual_sr,
            )

        # ✅ Always use the ACTUAL sample rate from the WAV header
        return TTSResult(
            audio=audio,
            sample_rate=actual_sr,
            duration_ms=len(audio) / actual_sr * 1000,
            engine_used="F5TTS",
            used_seed=meta.get("used_seed"),
            reference_voice_info=meta.get("reference_voice_info"),
        )

    # ================================================================== #
    #  Reference voice management
    # ================================================================== #

    def list_reference_voices(self, force_refresh: bool = False) -> Dict[str, ReferenceVoice]:
        return self.indicf5_client.list_reference_voices(force_refresh=force_refresh)

    def upload_reference_voice(
        self,
        filepath: str,
        name: str,
        author: str,
        content: str = "",
        model: str = "F5TTS",
    ) -> dict:
        return self.indicf5_client.upload_reference_voice(
            filepath=filepath, name=name, author=author,
            content=content, model=model,
        )

    def delete_reference_voice(self, voice_key: str) -> dict:
        return self.indicf5_client.delete_reference_voice(voice_key)

    def set_reference_voice(self, voice_key: str):
        """Change the default reference voice at runtime."""
        self._indicf5_reference_voice_key = voice_key
        self.indicf5_client.default_reference_voice_key = voice_key
        logger.info("indicf5_voice_changed", voice_key=voice_key)

    # ================================================================== #
    #  Prompt-tagged & batch synthesis
    # ================================================================== #

    def synthesize_prompt_tagged(self, text: str, **kwargs) -> dict:
        """Synthesize text containing <refvoice key='...'>...</refvoice> tags."""
        return self.indicf5_client.synthesize_prompt_tagged(text, **kwargs)

    def synthesize_batch(self, requests_list: List[Dict[str, Any]], **kwargs) -> dict:
        """Batch-synthesize multiple texts."""
        return self.indicf5_client.synthesize_batch(requests_list, **kwargs)

    # ================================================================== #
    #  Podcast
    # ================================================================== #

    def generate_podcast(self, **kwargs) -> dict:
        return self.indicf5_client.generate_podcast(**kwargs)

    # ================================================================== #
    #  Properties & cleanup
    # ================================================================== #

    @property
    def available_engines(self) -> List[str]:
        return ["F5TTS"]

    @property
    def active_engine(self) -> str:
        return "F5TTS"

    def cleanup(self):
        self.indicf5_client.close()
        self._is_initialized = False