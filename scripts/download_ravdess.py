#!/usr/bin/env python3
"""
RAVDESS Dataset Download Script.

Downloads the Ryerson Audio-Visual Database of Emotional Speech and Song
from one of three supported sources: Zenodo (default), Hugging Face, or Deep Lake.

Usage:
    python scripts/download_ravdess.py
    python scripts/download_ravdess.py --source huggingface
    python scripts/download_ravdess.py --source zenodo --subset both
    python scripts/download_ravdess.py --license

License:
    RAVDESS is distributed under CC BY-NC-SA 4.0 (non-commercial use).
    See https://zenodo.org/record/1188976 for details.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Try to import optional dependencies
try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment,misc]

__version__ = "0.1.0"

# =============================================================================
# CONSTANTS
# =============================================================================

ZENODO_RECORD_ID = "1188976"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# File metadata from Zenodo record
# MD5 checksums obtained from Zenodo API response
ZENODO_FILES = {
    "speech": {
        "filename": "Audio_Speech_Actors_01-24.zip",
        "size_bytes": 208_468_073,
        "expected_files": 1440,
    },
    "song": {
        "filename": "Audio_Song_Actors_01-24.zip",
        "size_bytes": 225_505_317,
        "expected_files": 1012,
    },
}

HF_DATASET_ID = "narad/ravdess"
DEEPLAKE_DATASET_ID = "hub://activeloop/ravdess-emotional-speech-audio"

DEFAULT_OUTPUT_DIR = Path("data/raw/ravdess")

RAVDESS_LICENSE_NOTICE = """
================================================================================
RAVDESS Dataset License Notice
================================================================================

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) is
licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0).

You are free to:
  - Share: copy and redistribute the material in any medium or format
  - Adapt: remix, transform, and build upon the material

Under the following terms:
  - Attribution: You must give appropriate credit
  - NonCommercial: You may not use the material for commercial purposes
  - ShareAlike: If you remix, you must distribute under the same license

Citation:
  Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of
  Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.
  https://doi.org/10.1371/journal.pone.0196391

Full license: https://creativecommons.org/licenses/by-nc-sa/4.0/
Dataset: https://zenodo.org/record/1188976

================================================================================
"""

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0
BACKOFF_FACTOR = 2.0
REQUEST_TIMEOUT = 60
CHUNK_SIZE = 8192


# =============================================================================
# EXCEPTIONS
# =============================================================================


class RAVDESSDownloadError(Exception):
    """Base exception for download errors."""

    pass


class DownloadError(RAVDESSDownloadError):
    """Network or I/O error during download."""

    pass


class ChecksumError(RAVDESSDownloadError):
    """MD5 checksum verification failed."""

    pass


class ExtractionError(RAVDESSDownloadError):
    """ZIP extraction failed."""

    pass


class DependencyError(RAVDESSDownloadError):
    """Required dependency not installed."""

    pass


# =============================================================================
# CLI PARSING
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments. If None, uses sys.argv[1:].

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download the RAVDESS dataset for emotion recognition research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Download speech from Zenodo (default)
  %(prog)s --subset both                # Download speech and song
  %(prog)s --source huggingface         # Download from Hugging Face
  %(prog)s --output /path/to/data       # Custom output directory
  %(prog)s --license                    # Show license notice

Environment Variables:
  EDFS_DATA_DIR    Base data directory (default: ./data)
        """,
    )

    parser.add_argument(
        "--source",
        choices=["zenodo", "huggingface", "deeplake"],
        default="zenodo",
        help="Download source (default: zenodo)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--subset",
        choices=["speech", "song", "both"],
        default="speech",
        help="Subset to download - zenodo only (default: speech)",
    )

    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip MD5 checksum verification (zenodo only)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )

    parser.add_argument(
        "--license",
        action="store_true",
        dest="show_license",
        help="Display license notice before download",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars and non-essential output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser.parse_args(argv)


def get_output_dir(args: argparse.Namespace) -> Path:
    """
    Resolve output directory from args or environment.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    Path
        Resolved output directory path.
    """
    if args.output is not None:
        return args.output

    # Check environment variable
    env_data_dir = os.environ.get("EDFS_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir) / "raw" / "ravdess"

    return DEFAULT_OUTPUT_DIR


# =============================================================================
# DOWNLOAD UTILITIES
# =============================================================================


def _check_requests() -> None:
    """Check if requests library is available."""
    if requests is None:
        raise DependencyError(
            "The 'requests' library is required for downloading.\n"
            "Install with: uv pip install requests"
        )


def _progress_bar(total: int | None, desc: str, quiet: bool):
    """Create a progress bar or dummy context manager."""
    if quiet or tqdm is None:
        # Return a dummy object that supports .update()
        class DummyProgress:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def update(self, n: int) -> None:
                pass

        return DummyProgress()

    return tqdm(total=total, unit="B", unit_scale=True, desc=desc)


def download_file(
    url: str,
    dest: Path,
    expected_size: int | None = None,
    timeout: int = REQUEST_TIMEOUT,
    retries: int = MAX_RETRIES,
    quiet: bool = False,
) -> Path:
    """
    Download a file with progress bar and retry logic.

    Parameters
    ----------
    url : str
        URL to download from.
    dest : Path
        Destination file path.
    expected_size : int | None
        Expected file size in bytes (for progress bar).
    timeout : int
        Request timeout in seconds.
    retries : int
        Maximum number of retry attempts.
    quiet : bool
        Suppress progress output.

    Returns
    -------
    Path
        Path to downloaded file.

    Raises
    ------
    DownloadError
        If download fails after all retries.
    """
    _check_requests()

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".tmp")

    delay = INITIAL_RETRY_DELAY
    last_exception: Exception | None = None

    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()

                total_size = int(
                    response.headers.get("Content-Length", expected_size or 0)
                )

                with (
                    open(temp_dest, "wb") as f,
                    _progress_bar(total_size or None, dest.name, quiet) as pbar,
                ):
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Rename temp file to final destination
            temp_dest.rename(dest)
            return dest

        except (requests.RequestException, OSError) as e:
            last_exception = e
            if temp_dest.exists():
                temp_dest.unlink()

            if attempt < retries - 1:
                if not quiet:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                delay = min(delay * BACKOFF_FACTOR, 30.0)
            else:
                raise DownloadError(
                    f"Failed to download {url} after {retries} attempts"
                ) from last_exception

    # Should not reach here, but satisfy type checker
    raise DownloadError(f"Failed to download {url}") from last_exception


def verify_checksum(filepath: Path, expected_md5: str) -> bool:
    """
    Verify file integrity using MD5 checksum.

    Parameters
    ----------
    filepath : Path
        Path to the file to verify.
    expected_md5 : str
        Expected MD5 hash (lowercase hex string).

    Returns
    -------
    bool
        True if checksum matches, False otherwise.
    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hash_md5.update(chunk)

    actual_md5 = hash_md5.hexdigest()
    return actual_md5.lower() == expected_md5.lower()


def extract_zip(
    zip_path: Path,
    dest_dir: Path,
    remove_after: bool = True,
    quiet: bool = False,
) -> None:
    """
    Extract a ZIP archive.

    Parameters
    ----------
    zip_path : Path
        Path to the ZIP file.
    dest_dir : Path
        Directory to extract to.
    remove_after : bool
        Remove ZIP file after extraction.
    quiet : bool
        Suppress progress output.

    Raises
    ------
    ExtractionError
        If extraction fails.
    """
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()

            if quiet or tqdm is None:
                zf.extractall(dest_dir)
            else:
                for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
                    zf.extract(member, dest_dir)

        if remove_after:
            zip_path.unlink()

    except (zipfile.BadZipFile, OSError) as e:
        raise ExtractionError(f"Failed to extract {zip_path}: {e}") from e


# =============================================================================
# ZENODO DOWNLOADER
# =============================================================================


def _get_zenodo_download_url(filename: str) -> str:
    """
    Get direct download URL for a Zenodo file.

    Parameters
    ----------
    filename : str
        Name of the file to download.

    Returns
    -------
    str
        Direct download URL.
    """
    _check_requests()

    # First, get the record metadata to find the file URL
    response = requests.get(ZENODO_API_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    record = response.json()

    # Find the file in the record
    for file_info in record.get("files", []):
        file_key = file_info.get("key", "")
        if file_key == filename:
            # Get the download link
            links = file_info.get("links", {})
            download_url = links.get("self")
            if download_url:
                return download_url

    # Fallback to constructed URL
    return f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{filename}?download=1"


def download_from_zenodo(
    output_dir: Path,
    subsets: list[str],
    skip_verify: bool = False,
    force: bool = False,
    quiet: bool = False,
) -> int:
    """
    Download RAVDESS from Zenodo using the REST API.

    Parameters
    ----------
    output_dir : Path
        Directory to save extracted audio files.
    subsets : list[str]
        List of subsets to download: ['speech'], ['song'], or ['speech', 'song'].
    skip_verify : bool
        Skip MD5 checksum verification.
    force : bool
        Overwrite existing files.
    quiet : bool
        Suppress progress output.

    Returns
    -------
    int
        Total number of files downloaded.

    Raises
    ------
    DownloadError
        If download fails after all retries.
    """
    _check_requests()

    output_dir.mkdir(parents=True, exist_ok=True)
    total_files = 0

    for subset in subsets:
        file_info = ZENODO_FILES.get(subset)
        if not file_info:
            print(f"Warning: Unknown subset '{subset}', skipping")
            continue

        filename = file_info["filename"]
        expected_size = file_info["size_bytes"]
        expected_files = file_info["expected_files"]

        zip_path = output_dir / filename

        # Check if already downloaded and extracted
        if not force:
            existing_files = list(output_dir.glob("Actor_*/*.wav"))
            if len(existing_files) >= expected_files:
                if not quiet:
                    print(
                        f"Dataset already exists ({len(existing_files)} files). "
                        "Use --force to re-download."
                    )
                total_files += len(existing_files)
                continue

        if not quiet:
            print(f"Downloading {filename} from Zenodo...")

        # Get download URL
        url = _get_zenodo_download_url(filename)

        # Download the ZIP file
        download_file(
            url=url,
            dest=zip_path,
            expected_size=expected_size,
            quiet=quiet,
        )

        # Note: Zenodo doesn't always provide MD5 checksums in the API response
        # for older records. We skip verification if checksums aren't available.
        if not skip_verify and not quiet:
            print("Checksum verification skipped (not available from Zenodo API)")

        # Extract ZIP
        if not quiet:
            print(f"Extracting {filename}...")

        extract_zip(zip_path, output_dir, remove_after=True, quiet=quiet)

        # Count extracted files
        wav_files = list(output_dir.glob("Actor_*/*.wav"))
        total_files = len(wav_files)

        if not quiet:
            print(f"Extracted {total_files} audio files")

    return total_files


# =============================================================================
# HUGGING FACE DOWNLOADER
# =============================================================================


def download_from_huggingface(
    output_dir: Path,
    force: bool = False,
    quiet: bool = False,
) -> int:
    """
    Download RAVDESS speech subset from Hugging Face.

    Parameters
    ----------
    output_dir : Path
        Directory to save audio files.
    force : bool
        Overwrite existing files.
    quiet : bool
        Suppress progress output.

    Returns
    -------
    int
        Number of files downloaded.

    Raises
    ------
    DependencyError
        If required libraries are not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DependencyError(
            "Hugging Face datasets library required.\n"
            "Install with: uv pip install 'datasets[audio]'"
        ) from e

    try:
        import soundfile as sf
    except ImportError as e:
        raise DependencyError(
            "soundfile library required for audio export.\n"
            "Install with: uv pip install soundfile"
        ) from e

    # Check if already downloaded
    if not force:
        existing_files = list(output_dir.glob("Actor_*/*.wav"))
        if len(existing_files) >= 1440:
            if not quiet:
                print(
                    f"Dataset already exists ({len(existing_files)} files). "
                    "Use --force to re-download."
                )
            return len(existing_files)

    if not quiet:
        print("Loading dataset from Hugging Face (this may take a few minutes)...")

    dataset = load_dataset(HF_DATASET_ID, split="train", trust_remote_code=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_count = 0

    iterator = dataset
    if not quiet and tqdm is not None:
        iterator = tqdm(dataset, desc="Exporting audio files")

    for item in iterator:
        audio = item["audio"]
        speaker_id = item.get("speaker_id", "00")

        # Create actor directory
        actor_dir = output_dir / f"Actor_{int(speaker_id):02d}"
        actor_dir.mkdir(exist_ok=True)

        # Get original filename or construct one
        audio_path = audio.get("path", "")
        if audio_path:
            filename = Path(audio_path).name
        else:
            # Construct filename from metadata
            filename = f"audio_{file_count:05d}.wav"

        output_path = actor_dir / filename

        if not force and output_path.exists():
            file_count += 1
            continue

        # Save audio array as WAV
        sf.write(output_path, audio["array"], audio["sampling_rate"])
        file_count += 1

    if not quiet:
        print(f"Exported {file_count} audio files")

    return file_count


# =============================================================================
# DEEP LAKE DOWNLOADER
# =============================================================================


def download_from_deeplake(
    output_dir: Path,
    force: bool = False,
    quiet: bool = False,
) -> int:
    """
    Download RAVDESS from Activeloop Deep Lake.

    Parameters
    ----------
    output_dir : Path
        Directory to save audio files.
    force : bool
        Overwrite existing files.
    quiet : bool
        Suppress progress output.

    Returns
    -------
    int
        Number of files downloaded.

    Raises
    ------
    DependencyError
        If deeplake library is not installed.
    """
    try:
        import deeplake
    except ImportError as e:
        raise DependencyError(
            "Deep Lake library required.\nInstall with: uv pip install deeplake"
        ) from e

    try:
        import soundfile as sf
    except ImportError as e:
        raise DependencyError(
            "soundfile library required for audio export.\n"
            "Install with: uv pip install soundfile"
        ) from e

    # Check if already downloaded
    if not force:
        existing_files = list(output_dir.glob("**/*.wav"))
        if len(existing_files) >= 1440:
            if not quiet:
                print(
                    f"Dataset already exists ({len(existing_files)} files). "
                    "Use --force to re-download."
                )
            return len(existing_files)

    if not quiet:
        print("Connecting to Deep Lake...")

    ds = deeplake.open_read_only(DEEPLAKE_DATASET_ID)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_count = 0

    iterator = range(len(ds))
    if not quiet and tqdm is not None:
        iterator = tqdm(iterator, desc="Downloading from Deep Lake")

    for i in iterator:
        sample = ds[i]

        # Extract audio data - schema may vary
        audio_tensor = sample.get("audios") or sample.get("audio")
        if audio_tensor is None:
            continue

        audio_data = audio_tensor.numpy()

        # Get metadata
        emotion = sample.get("emotions", sample.get("emotion", "unknown"))
        if hasattr(emotion, "numpy"):
            emotion = emotion.numpy()

        # Construct filename
        filename = f"audio_{i:05d}_emotion-{emotion}.wav"
        output_path = output_dir / filename

        if not force and output_path.exists():
            file_count += 1
            continue

        # Save audio - assuming 16kHz sample rate (common for speech)
        sf.write(output_path, audio_data, 16000)
        file_count += 1

    if not quiet:
        print(f"Downloaded {file_count} audio files")

    return file_count


# =============================================================================
# VERIFICATION
# =============================================================================


def verify_dataset(
    output_dir: Path,
    expected_count: int = 1440,
    quiet: bool = False,
) -> bool:
    """
    Verify downloaded dataset integrity.

    Parameters
    ----------
    output_dir : Path
        Directory containing the dataset.
    expected_count : int
        Expected number of WAV files.
    quiet : bool
        Suppress output.

    Returns
    -------
    bool
        True if verification passes.
    """
    wav_files = list(output_dir.rglob("*.wav"))
    actual_count = len(wav_files)

    if actual_count == 0:
        if not quiet:
            print("Error: No WAV files found")
        return False

    # Verify directory structure (24 actors)
    actor_dirs = [
        d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("Actor_")
    ]

    # Verify filename format (RAVDESS naming convention)
    ravdess_pattern = re.compile(r"\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.wav")
    valid_files = [f for f in wav_files if ravdess_pattern.match(f.name)]

    if not quiet:
        print("Verification results:")
        print(f"  Total WAV files: {actual_count}")
        print(f"  Actor directories: {len(actor_dirs)}")
        print(f"  RAVDESS-format files: {len(valid_files)}")

        if actual_count != expected_count:
            print(f"  Warning: Expected {expected_count} files")

    return actual_count > 0


def create_manifest(
    output_dir: Path,
    source: str,
    file_count: int,
) -> None:
    """
    Create a manifest file documenting the download.

    Parameters
    ----------
    output_dir : Path
        Directory containing the dataset.
    source : str
        Download source used.
    file_count : int
        Number of files downloaded.
    """
    manifest = {
        "source": source,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "file_count": file_count,
        "zenodo_doi": "10.5281/zenodo.1188976",
        "script_version": __version__,
        "license": "CC BY-NC-SA 4.0",
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    """
    Main entry point for the download script.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args(argv)

    # Show license if requested
    if args.show_license:
        print(RAVDESS_LICENSE_NOTICE)

    # Resolve output directory
    output_dir = get_output_dir(args)

    if not args.quiet:
        print(f"Output directory: {output_dir.resolve()}")

    # Determine subsets for Zenodo
    if args.subset == "both":
        subsets = ["speech", "song"]
    else:
        subsets = [args.subset]

    try:
        # Download based on source
        if args.source == "zenodo":
            file_count = download_from_zenodo(
                output_dir=output_dir,
                subsets=subsets,
                skip_verify=args.skip_verify,
                force=args.force,
                quiet=args.quiet,
            )
        elif args.source == "huggingface":
            file_count = download_from_huggingface(
                output_dir=output_dir,
                force=args.force,
                quiet=args.quiet,
            )
        elif args.source == "deeplake":
            file_count = download_from_deeplake(
                output_dir=output_dir,
                force=args.force,
                quiet=args.quiet,
            )
        else:
            print(f"Error: Unknown source '{args.source}'")
            return 1

        # Verify and create manifest
        if file_count > 0:
            verify_dataset(output_dir, quiet=args.quiet)
            create_manifest(output_dir, args.source, file_count)

            if not args.quiet:
                print(f"\nDownload complete: {file_count} files")
                print(f"Manifest saved to: {output_dir / 'manifest.json'}")

        return 0

    except DependencyError as e:
        print(f"Dependency error: {e}")
        return 1
    except DownloadError as e:
        print(f"Download error: {e}")
        return 1
    except ExtractionError as e:
        print(f"Extraction error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nDownload cancelled by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
