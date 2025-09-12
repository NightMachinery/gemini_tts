#!/usr/bin/env python3

from pynight.common_icecream import ic

import argparse
import asyncio
import sys
from pathlib import Path

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
        description="A multi-speaker TTS script to convert a podcast script into audio using Gemini APIs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "input_paths", nargs="+", help="One or more input text/markdown file paths."
    )
    parser.add_argument(
        "-o",
        "--out",
        help="The base path for output files. Defaults to the first input file's name.",
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash-preview-tts",
        help="The Gemini TTS model to use for token counting and generation. (Default: %(default)s)",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=8000, #: 8192 is max supported by Flash TTS 2.5, and I am leaving some margin of error
        help="Max number of tokens per chunk. (Default: 8000)",
    )
    parser.add_argument(
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
        "--no-speakers",
        action="store_true",
        help="Disable multi-speaker mode entirely. Ignores --speakers.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel API calls to make. (Default: 1)",
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
        action="store_true",
        help="Remove intermediate chunk files after merging.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level. Use -v for basic info, -vv for detailed info, -vvv for debug info.",
    )

    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(args.input_paths[0]).with_suffix("")

    ##
    model = args.model
    if model.lower() == "g25":
        model = "gemini-2.5-pro-preview-tts"
    elif model.lower() == "flash":
        model = "gemini-2.5-flash-preview-tts"
    ##

    # Use dependency injection by creating a config object
    tts_config = TTSConfig(
        model=model,
        max_chunk_tokens=args.max_chunk_tokens,
        speakers=args.speakers,
        no_speakers=args.no_speakers,
        parallel=args.parallel,
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
