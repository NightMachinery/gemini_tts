"""Gemini TTS - Convert long documents to audio using Google's Gemini TTS API."""

from .tts_lib import TTSConfig, run_tts_pipeline, TTSResult, Chunk

__version__ = "0.1.0"
__all__ = ["TTSConfig", "run_tts_pipeline", "TTSResult", "Chunk"]
