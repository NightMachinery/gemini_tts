from pynight.common_icecream import ic

import asyncio
import os
import re
import struct
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiofiles
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm
from pynight.common_gemini_tts import GEMINI_VOICES

# A selection of diverse voices for automatic speaker assignment
DEFAULT_VOICES = list(GEMINI_VOICES.keys())

# --- Dataclasses for Configuration and Results ---


@dataclass(frozen=True)
class TTSConfig:
    """Bundles all operational configurations for the TTS pipeline."""

    model: str
    max_chunk_tokens: int
    speakers: str
    no_speakers: bool
    parallel: int
    retries: int
    retry_sleep: int
    cleanup_chunks: bool
    verbose: int = 0


@dataclass
class Chunk:
    """Represents a single chunk of text to be processed."""

    index: int
    text: str
    text_path: Path
    audio_path: Path
    status: str = "pending"
    error_message: Optional[str] = None


@dataclass
class TTSResult:
    """Represents the final output of the TTS process."""

    chunks: List[Chunk]
    final_audio_path: Optional[Path] = None
    success: bool = True
    message: str = "Processing complete."

    def get_failed_chunks(self) -> List[Chunk]:
        """Returns a list of chunks that failed processing."""
        return [chunk for chunk in self.chunks if chunk.status == "failed"]


# --- Core Logic ---


def _log(message: str, level: int, config_verbose: int):
    """Log a message if the verbosity level is high enough."""
    if config_verbose >= level:
        print(f"[V{level}] {message}")


def _read_and_join_files(input_paths: List[Path]) -> str:
    """Reads content from all input files and joins them with a separator."""
    contents = []
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input path is not a file: {path}")
        contents.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(contents)


def _normalize_speaker_labels(text: str) -> str:
    """Removes markdown formatting (e.g., bold, italics) from speaker labels."""
    return re.sub(r"([*_~`]){1,3}([^:*_~`\n]{1,25}):\1{1,3}", r"\2:", text)


def _determine_speakers(
    text: str, *, speaker_config: str
) -> tuple[Dict[str, str], Set[str]]:
    """
    Parses speaker configuration to map speaker names to voices.

    Returns:
        A tuple containing (speaker_voice_map, all_speaker_names).
    """
    speaker_voice_map: Dict[str, str] = {}

    if "auto:" in speaker_config:
        try:
            num_speakers = int(speaker_config.split(":")[1])
        except (ValueError, IndexError):
            raise ValueError(
                "Invalid auto speaker format. Use 'auto:N' where N is a number."
            )

        normalized_text = _normalize_speaker_labels(text)
        speaker_regex = re.compile(r"^([^:]{1,25}):", re.MULTILINE)
        found_speakers = [
            match.strip() for match in speaker_regex.findall(normalized_text)
        ]

        if not found_speakers:
            raise ValueError(
                "Auto speaker detection found no labels matching '^NAME:'."
            )

        speakers = [
            speaker for speaker, _ in Counter(found_speakers).most_common(num_speakers)
        ]
        _log(f"Automatically detected speakers: {speakers}", 1, 1)  # Always show speaker detection
    else:
        speakers = [part.strip() for part in speaker_config.split(",")]

    for i, speaker_part in enumerate(speakers):
        if ":" in speaker_part:
            speaker, voice = [p.strip() for p in speaker_part.split(":", 1)]
            speaker_voice_map[speaker] = voice
        else:
            speaker = speaker_part
            speaker_voice_map[speaker] = DEFAULT_VOICES[i % len(DEFAULT_VOICES)]

    return speaker_voice_map, set(speaker_voice_map.keys())


async def _chunk_text(
    text: str, *, speakers: Set[str], max_tokens: int, client, model: str, verbose: int = 0
) -> List[str]:
    """Breaks text into chunks using a tokenizer, preferring speaker boundaries."""
    chunks: List[str] = []
    lines = text.split("\n")
    current_line_idx = 0
    
    speaker_line_regex = re.compile(
        f"^({'|'.join(re.escape(s) for s in speakers)}):", re.MULTILINE
    ) if speakers else None

    while current_line_idx < len(lines):
        # Start with a single line and expand until we hit the token limit
        chunk_lines = [lines[current_line_idx]]
        end_idx = current_line_idx + 1
        
        # Expand chunk until we hit token limit
        while end_idx < len(lines):
            potential_chunk = "\n".join(chunk_lines + [lines[end_idx]])
            token_count_result = await client.aio.models.count_tokens(
                model=model, contents=potential_chunk
            )
            
            if token_count_result.total_tokens > max_tokens:
                break
                
            chunk_lines.append(lines[end_idx])
            end_idx += 1
        
        # If we have more than one line and speaker detection is enabled,
        # try to find a better breaking point by searching backward for speaker lines
        if len(chunk_lines) > 1 and speaker_line_regex and end_idx < len(lines):
            # Search backward from the end for the last speaker line
            for i in range(len(chunk_lines) - 1, 0, -1):  # Don't include first line in search
                if speaker_line_regex.match(chunk_lines[i]):
                    # Found a speaker line, split here
                    speaker_line = chunk_lines[i]
                    chunk_lines = chunk_lines[:i]
                    end_idx = current_line_idx + i
                    _log(f"Adjusted chunk boundary at speaker line: '{speaker_line[:50]}...'", 3, verbose)
                    break
        
        chunk_text = "\n".join(chunk_lines)
        if chunk_text.strip():
            chunks.append(chunk_text)
            # Get actual token count for logging
            final_token_result = await client.aio.models.count_tokens(model=model, contents=chunk_text)
            _log(f"Created chunk {len(chunks)} with {final_token_result.total_tokens} tokens", 3, verbose)
        
        current_line_idx = end_idx

    _log(f"Chunking complete: {len(chunks)} total chunks", 2, verbose)
    return chunks


# --- Audio and File Handling ---


def _convert_to_wav(audio_data: bytes) -> bytes:
    """Generates a valid WAV file header for raw L16 audio data from the API."""
    bits_per_sample, sample_rate, num_channels = 16, 24000, 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def _merge_audio_files(chunks: List[Chunk], *, final_path: Path, verbose: int = 0) -> bool:
    """Merges all chunk .wav files into a single .mp3 file using ffmpeg."""
    _log("Merging audio chunks into final MP3 file...", 1, verbose)
    successful_chunks = [c for c in chunks if c.status == "success"]
    _log(f"Merging {len(successful_chunks)} successful chunks into {final_path}", 2, verbose)
    list_path = final_path.with_suffix(".txt")
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                if chunk.status == "success":
                    f.write(f"file '{chunk.audio_path.resolve()}'\n")

        command = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path.resolve()),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "3",
            "-y",
            str(final_path.resolve()),
        ]
        subprocess.run(
            command, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        _log(f"Successfully created final audio file: {final_path}", 1, verbose)
        return True
    except FileNotFoundError:
        print(
            "ERROR: ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH."
        )
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffmpeg failed to merge files.\nFFmpeg stderr:\n{e.stderr}")
        return False
    finally:
        if list_path.exists():
            os.remove(list_path)


def _cleanup_chunk_files(chunks: List[Chunk], verbose: int = 0):
    """Removes intermediate text and audio files for all provided chunks."""
    _log(f"Cleaning up {len(chunks)} intermediate chunk files...", 1, verbose)
    for chunk in chunks:
        if chunk.text_path.exists():
            os.remove(chunk.text_path)
        if chunk.audio_path.exists():
            os.remove(chunk.audio_path)


# --- Gemini API Interaction ---


def _build_speech_config(
    *, no_speakers: bool, speaker_voice_map: Dict[str, str]
) -> types.SpeechConfig:
    """Constructs the appropriate speech configuration for the API call."""
    if no_speakers:
        return types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=DEFAULT_VOICES[0]
                )
            )
        )
    return types.SpeechConfig(
        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
            speaker_voice_configs=[
                types.SpeakerVoiceConfig(
                    speaker=speaker,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    ),
                )
                for speaker, voice in speaker_voice_map.items()
            ]
        )
    )


async def _process_chunk(
    chunk: Chunk,
    *,
    progress_bar: tqdm,
    semaphore: asyncio.Semaphore,
    config: TTSConfig,
    client,
    speaker_voice_map: Dict[str, str],
):
    """Processes a single text chunk, handling caching, API calls, and retries."""
    async with semaphore:
        if chunk.audio_path.exists():
            # Check if the cached text matches current text (chunk.text_path should already exist)
            try:
                async with aiofiles.open(chunk.text_path, "r", encoding="utf-8") as f:
                    cached_text = await f.read()
                    if cached_text == chunk.text:
                        chunk.status = "skipped"
                        _log(f"Chunk {chunk.index}: Using cached audio", 3, config.verbose)
                        progress_bar.update(1)
                        return
            except IOError:
                pass  # File is unreadable, proceed to regenerate.

        # Text chunk should already be saved, but verify it exists
        if not chunk.text_path.exists():
            _log(f"Chunk {chunk.index}: Text file missing, recreating", 2, config.verbose)
            async with aiofiles.open(chunk.text_path, "w", encoding="utf-8") as f:
                await f.write(chunk.text)

        speech_config = _build_speech_config(
            no_speakers=config.no_speakers, speaker_voice_map=speaker_voice_map
        )
        generation_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"], speech_config=speech_config
        )

        for attempt in range(config.retries):
            try:
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=chunk.text)],
                    )
                ]
                response = await client.aio.models.generate_content(
                    model=config.model,
                    contents=contents,
                    config=generation_config,
                )
                audio_data = response.candidates[0].content.parts[0].inline_data.data
                wav_data = _convert_to_wav(audio_data)
                async with aiofiles.open(chunk.audio_path, "wb") as f:
                    await f.write(wav_data)
                chunk.status = "success"
                _log(f"Chunk {chunk.index}: Generated audio successfully", 3, config.verbose)
                break
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{config.retries} failed: {e}"
                _log(f"Chunk {chunk.index}: {error_msg}", 2, config.verbose)
                if attempt < config.retries - 1:
                    await asyncio.sleep(config.retry_sleep)
                else:
                    chunk.status = "failed"
                    chunk.error_message = str(e)
        progress_bar.update(1)


# --- Main Pipeline Orchestrator ---


async def run_tts_pipeline(
    input_paths: List[Path], out_path: Path, *, config: TTSConfig
) -> TTSResult:
    """Orchestrates the entire TTS process from text input to final audio file."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return TTSResult(
            chunks=[],
            success=False,
            message="GEMINI_API_KEY environment variable not set.",
        )

    try:
        client = genai.Client(api_key=api_key)
        _log(f"Initialized Gemini client with model: {config.model}", 1, config.verbose)
        
        full_text = _read_and_join_files(input_paths)
        _log(f"Read {len(input_paths)} input file(s), total characters: {len(full_text)}", 1, config.verbose)
        speaker_voice_map, all_speakers = {}, set()
        if not config.no_speakers:
            speaker_voice_map, all_speakers = _determine_speakers(
                full_text, speaker_config=config.speakers
            )

        _log("Chunking text based on token limits...", 1, config.verbose)
        # Get total token count before chunking
        total_token_result = await client.aio.models.count_tokens(
            model=config.model, contents=full_text
        )
        total_tokens = total_token_result.total_tokens
        _log(f"Total input tokens: {total_tokens}", 1, config.verbose)
        
        text_chunks = await _chunk_text(
            full_text,
            speakers=all_speakers,
            max_tokens=config.max_chunk_tokens,
            client=client,
            model=config.model,
            verbose=config.verbose,
        )
        if not text_chunks:
            return TTSResult(
                chunks=[],
                success=False,
                message="No text chunks could be created from input.",
            )
        
        _log(f"Created {len(text_chunks)} chunks (max {config.max_chunk_tokens} tokens each)", 1, config.verbose)

        chunks = []
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save all text chunks before beginning TTS
        _log("Saving all text chunks to disk...", 1, config.verbose)
        for i, text_chunk in enumerate(text_chunks):
            base_name = f"{out_path.stem}_{i}"
            chunk = Chunk(
                index=i,
                text=text_chunk,
                text_path=out_path.parent / f"tmp_{base_name}.md",
                audio_path=out_path.parent / f"{base_name}.wav",
            )
            chunks.append(chunk)
            
            # Save the text chunk to disk immediately
            async with aiofiles.open(chunk.text_path, 'w', encoding='utf-8') as f:
                await f.write(text_chunk)
            _log(f"Saved chunk {i+1}/{len(text_chunks)}: {chunk.text_path}", 2, config.verbose)

        _log(f"Processing {len(chunks)} chunks with {config.parallel} parallel requests", 1, config.verbose)
        _log(f"Configuration: max_tokens_per_chunk={config.max_chunk_tokens}, retries={config.retries}, retry_sleep={config.retry_sleep}s", 2, config.verbose)
        semaphore = asyncio.Semaphore(config.parallel)
        progress = tqdm(total=len(chunks), desc="Generating Audio Chunks")

        tasks = [
            _process_chunk(
                c,
                progress_bar=progress,
                semaphore=semaphore,
                config=config,
                client=client,
                speaker_voice_map=speaker_voice_map,
            )
            for c in chunks
        ]
        await asyncio.gather(*tasks)
        progress.close()

        result = TTSResult(chunks=chunks)
        if failed_chunks := result.get_failed_chunks():
            result.success = False
            result.message = f"{len(failed_chunks)} chunk(s) failed to process. Halting before merge."
            print(f"\nERROR: {result.message}")
            for failed in failed_chunks:
                print(f"  - Chunk {failed.index}: {failed.error_message}")
            return result

        final_mp3_path = out_path.with_suffix(".mp3")
        if not _merge_audio_files(result.chunks, final_path=final_mp3_path, verbose=config.verbose):
            result.success = False
            result.message = "Failed to merge audio chunks with ffmpeg."
            return result

        result.final_audio_path = final_mp3_path
        if config.cleanup_chunks:
            _cleanup_chunk_files(result.chunks, verbose=config.verbose)

        return result
    except Exception as e:
        return TTSResult(
            chunks=[], success=False, message=f"An unexpected error occurred: {e}"
        )
