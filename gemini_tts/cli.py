#!/usr/bin/env python3

from pynight.common_icecream import ic

import argparse
import asyncio
import sys
from pathlib import Path
import os
from typing import List

try:
    from .tts_lib import TTSConfig, run_tts_pipeline
except ImportError:
    print(
        "Error: Could not import 'tts_lib'. Make sure 'tts_lib.py' is in the same directory."
    )
    sys.exit(1)


def main():
    """Parses command-line arguments and runs the TTS pipeline."""
    parser = argparse.ArgumentParser(
        description="A multi-speaker TTS script to convert a podcast script into audio using the Gemini API.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "input_paths", nargs="+", help="One or more input text/markdown file paths."
    )
    parser.add_argument(
        "-o",
        "--out",
        help=(
            "Output base path or directory. If it ends with '/', the first input file's basename"
            " (sans extension) is appended. Defaults to first input file's name."
        ),
        default=None,
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Path to a file containing Google Gemini API keys, one per line.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gemini-2.5-flash-preview-tts",
        help="The Gemini TTS model to use for token counting and generation. (Default: %(default)s)",
    )
    max_chunk_tokens_default = 7500
    #: 8192 is max supported by Flash TTS 2.5. Leave more margin of error if you want to use (old) offline token counters which are inaccurate. Even the online token counter needs some margin of error.
    parser.add_argument(
        "-t",
        "--max-chunk-tokens",
        type=int,
        default=max_chunk_tokens_default,
        help=f"Max number of tokens per chunk. (Default: {max_chunk_tokens_default})",
    )
    parser.add_argument(
        "-s",
        "--speakers",
        type=str,
        default="auto:2",
        help="""Speaker configuration.
Examples:
  'auto:2'              - (Default) Auto-detect the 2 most frequent speakers.
  'Host A,Host B'       - Explicitly name two speakers.
  'Host A:Zephyr,HostB' - Map a specific voice to a speaker.""",
    )
    parser.add_argument(
        "--multi-speakers",
        action=argparse.BooleanOptionalAction,
        dest="multi_speakers_p",
        default=True,
        help="Enable/disable multi-speaker mode. Use --no-multi-speakers to disable.",
    )
    parser.add_argument(
        "--hash-voices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include speaker-voice mapping in content hash for better cache invalidation. Use --no-hash-voices to disable.",
    )
    parser.add_argument(
        "--chunk-filename-include-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include 8-char hash in chunk filenames for easy identification. Use --no-chunk-filename-include-hash to disable.",
    )
    parser.add_argument(
        "--parallel",
        type=str,
        default="auto",
        help="Number of parallel API calls. Use 'auto' to set based on number of API keys. (Default: 1)",
    )
    parser.add_argument(
        "--parallel-per-key",
        type=int,
        default=1,
        help="Number of parallel requests allowed for a single API key. (Default: 1)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for a failed API call on a chunk. (Default: 3)",
    )
    parser.add_argument(
        "--retry-sleep",
        type=int,
        default=65,
        help="Seconds to wait between retries. (Default: 65, for API rate limits)",
    )
    parser.add_argument(
        "--cleanup-chunks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove intermediate chunk files after merging. Use --cleanup-chunks to enable.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level. Use -v for basic info, -vv for detailed info, -vvv for debug info.",
    )

    args = parser.parse_args()

    # --- API Key Management ---
    api_keys: List[str] = []
    if args.api_key_file:
        try:
            with open(args.api_key_file, "r") as f:
                api_keys = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: API key file not found at {args.api_key_file}")
            sys.exit(1)

    if not api_keys:
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            api_keys.append(env_key)

    if not api_keys:
        print(
            "Error: No API key provided. Use --api-key-file or set the GEMINI_API_KEY environment variable."
        )
        sys.exit(1)

    # --- Parallelism Configuration ---
    if args.parallel.lower() == "auto":
        parallel_count = len(api_keys) * args.parallel_per_key
    else:
        try:
            parallel_count = int(args.parallel)
            if parallel_count < 1:
                raise ValueError
        except ValueError:
            print(f"Error: --parallel must be a positive integer or 'auto'.")
            sys.exit(1)

    # Determine output base path. If --out ends with '/', treat it as a directory
    # and append the first input file's basename (sans extension).
    if args.out:
        out_arg: str = str(args.out)
        if out_arg.endswith("/") or (os.sep != "/" and out_arg.endswith(os.sep)):
            first_base = Path(args.input_paths[0]).with_suffix("").name
            out_path = Path(out_arg) / first_base
        else:
            out_path = Path(out_arg)
    else:
        out_path = Path(args.input_paths[0]).with_suffix("")

    ##
    model = args.model
    if model.lower() == "g25":
        model = "gemini-2.5-pro-preview-tts"
    elif model.lower() == "flash":
        model = "gemini-2.5-flash-preview-tts"
    ##

    # Use dependency injection by creating a config object
    tts_config = TTSConfig(
        api_keys=api_keys,
        model=model,
        max_chunk_tokens=args.max_chunk_tokens,
        speakers=args.speakers,
        speakers_enabled=args.multi_speakers_p,
        hash_voices=args.hash_voices,
        chunk_filename_include_hash=args.chunk_filename_include_hash,
        parallel=parallel_count,
        parallel_per_key=args.parallel_per_key,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        cleanup_chunks=args.cleanup_chunks,
        verbose=args.verbose,
    )

    # Run the main async pipeline
    result = asyncio.run(
        run_tts_pipeline(
            [Path(p) for p in args.input_paths],
            out_path,
            config=tts_config,
        )
    )

    print("-" * 50)
    if result.success:
        print(f"✅ Success! Final audio file created at: {result.final_audio_path}")
    else:
        print(f"❌ Failure: {result.message}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure you have the required libraries installed:
    # pip install google-generativeai aiofiles tqdm
    # And ffmpeg installed on your system.
    main()
