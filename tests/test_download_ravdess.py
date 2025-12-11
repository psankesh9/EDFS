"""Tests for RAVDESS download script."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from download_ravdess import (
    DEFAULT_OUTPUT_DIR,
    DependencyError,
    DownloadError,
    get_output_dir,
    parse_args,
    verify_checksum,
    verify_dataset,
)


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_default_arguments(self) -> None:
        """Test default argument values."""
        args = parse_args([])
        assert args.source == "zenodo"
        assert args.subset == "speech"
        assert args.output is None
        assert args.skip_verify is False
        assert args.force is False
        assert args.show_license is False
        assert args.quiet is False

    def test_zenodo_source_explicit(self) -> None:
        """Test explicit zenodo source selection."""
        args = parse_args(["--source", "zenodo"])
        assert args.source == "zenodo"

    def test_huggingface_source(self) -> None:
        """Test Hugging Face source selection."""
        args = parse_args(["--source", "huggingface"])
        assert args.source == "huggingface"

    def test_deeplake_source(self) -> None:
        """Test Deep Lake source selection."""
        args = parse_args(["--source", "deeplake"])
        assert args.source == "deeplake"

    def test_subset_both(self) -> None:
        """Test both subset selection."""
        args = parse_args(["--subset", "both"])
        assert args.subset == "both"

    def test_subset_song(self) -> None:
        """Test song subset selection."""
        args = parse_args(["--subset", "song"])
        assert args.subset == "song"

    def test_custom_output(self) -> None:
        """Test custom output directory."""
        args = parse_args(["--output", "/custom/path"])
        assert args.output == Path("/custom/path")

    def test_skip_verify_flag(self) -> None:
        """Test skip-verify flag."""
        args = parse_args(["--skip-verify"])
        assert args.skip_verify is True

    def test_force_flag(self) -> None:
        """Test force flag."""
        args = parse_args(["--force"])
        assert args.force is True

    def test_license_flag(self) -> None:
        """Test license flag."""
        args = parse_args(["--license"])
        assert args.show_license is True

    def test_quiet_flag(self) -> None:
        """Test quiet flag."""
        args = parse_args(["--quiet"])
        assert args.quiet is True

    def test_combined_flags(self) -> None:
        """Test multiple flags combined."""
        args = parse_args(
            [
                "--source",
                "zenodo",
                "--subset",
                "both",
                "--output",
                "/data",
                "--force",
                "--quiet",
            ]
        )
        assert args.source == "zenodo"
        assert args.subset == "both"
        assert args.output == Path("/data")
        assert args.force is True
        assert args.quiet is True


class TestGetOutputDir:
    """Tests for output directory resolution."""

    def test_default_path(self) -> None:
        """Test default output path."""
        args = parse_args([])
        output_dir = get_output_dir(args)
        assert output_dir == DEFAULT_OUTPUT_DIR

    def test_explicit_output_path(self) -> None:
        """Test explicit output path from args."""
        args = parse_args(["--output", "/custom/path"])
        output_dir = get_output_dir(args)
        assert output_dir == Path("/custom/path")

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test EDFS_DATA_DIR environment variable override."""
        monkeypatch.setenv("EDFS_DATA_DIR", "/env/data")
        args = parse_args([])
        output_dir = get_output_dir(args)
        assert output_dir == Path("/env/data/raw/ravdess")

    def test_explicit_takes_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that explicit --output takes precedence over env var."""
        monkeypatch.setenv("EDFS_DATA_DIR", "/env/data")
        args = parse_args(["--output", "/explicit/path"])
        output_dir = get_output_dir(args)
        assert output_dir == Path("/explicit/path")


class TestVerifyChecksum:
    """Tests for MD5 checksum verification."""

    def test_verify_checksum_valid(self, tmp_path: Path) -> None:
        """Test checksum verification with valid file."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")
        # MD5 of "test content" = 9473fdd0d880a43c21b7778d34872157
        expected_md5 = "9473fdd0d880a43c21b7778d34872157"
        assert verify_checksum(test_file, expected_md5) is True

    def test_verify_checksum_invalid(self, tmp_path: Path) -> None:
        """Test checksum verification with corrupted file."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"different content")
        expected_md5 = "9473fdd0d880a43c21b7778d34872157"
        assert verify_checksum(test_file, expected_md5) is False

    def test_verify_checksum_case_insensitive(self, tmp_path: Path) -> None:
        """Test checksum verification is case-insensitive."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")
        # Uppercase MD5
        expected_md5 = "9473FDD0D880A43C21B7778D34872157"
        assert verify_checksum(test_file, expected_md5) is True

    def test_verify_checksum_empty_file(self, tmp_path: Path) -> None:
        """Test checksum verification with empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")
        # MD5 of empty string = d41d8cd98f00b204e9800998ecf8427e
        expected_md5 = "d41d8cd98f00b204e9800998ecf8427e"
        assert verify_checksum(test_file, expected_md5) is True


class TestVerifyDataset:
    """Tests for post-download verification."""

    def test_verify_dataset_correct_count(self, tmp_path: Path) -> None:
        """Test verification with correct file count."""
        # Create mock directory structure with 1440 files (60 per actor)
        for i in range(1, 25):
            actor_dir = tmp_path / f"Actor_{i:02d}"
            actor_dir.mkdir()
            for j in range(60):
                # RAVDESS filename format: MM-VV-EE-II-SS-RR-AA.wav
                filename = f"03-01-01-01-01-{j % 60 + 1:02d}-{i:02d}.wav"
                (actor_dir / filename).touch()

        assert verify_dataset(tmp_path, expected_count=1440, quiet=True) is True

    def test_verify_dataset_missing_files(self, tmp_path: Path) -> None:
        """Test verification with missing files."""
        # Create only 10 files
        actor_dir = tmp_path / "Actor_01"
        actor_dir.mkdir()
        for i in range(10):
            (actor_dir / f"03-01-01-01-01-{i:02d}-01.wav").touch()

        # Should still return True (has files), but with warning
        assert verify_dataset(tmp_path, expected_count=1440, quiet=True) is True

    def test_verify_dataset_empty_directory(self, tmp_path: Path) -> None:
        """Test verification with empty directory."""
        assert verify_dataset(tmp_path, expected_count=1440, quiet=True) is False

    def test_verify_dataset_no_actor_dirs(self, tmp_path: Path) -> None:
        """Test verification with files but no actor directories."""
        # Create files directly in output dir (wrong structure)
        (tmp_path / "audio_001.wav").touch()
        (tmp_path / "audio_002.wav").touch()

        # Should return True (has wav files)
        assert verify_dataset(tmp_path, expected_count=1440, quiet=True) is True


class TestDownloadFile:
    """Tests for file download functionality."""

    def test_download_file_requires_requests(self) -> None:
        """Test that download_file raises error when requests not installed."""
        from download_ravdess import _check_requests

        # This test passes if requests is installed (normal case)
        # The actual test of missing dependency would require uninstalling requests
        _check_requests()  # Should not raise

    @patch("download_ravdess.requests")
    def test_download_file_creates_parent_dirs(
        self, mock_requests: MagicMock, tmp_path: Path
    ) -> None:
        """Test that download creates parent directories."""
        from download_ravdess import download_file

        # Setup mock response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.iter_content.return_value = [b"x" * 100]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_requests.get.return_value = mock_response

        dest = tmp_path / "nested" / "dir" / "file.zip"
        download_file("http://example.com/file.zip", dest, quiet=True)

        assert dest.parent.exists()


class TestMainFunction:
    """Tests for main entry point."""

    def test_main_shows_help(self, capsys: pytest.CaptureFixture) -> None:
        """Test that --help works."""
        from download_ravdess import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "RAVDESS" in captured.out
        assert "--source" in captured.out

    def test_main_shows_version(self, capsys: pytest.CaptureFixture) -> None:
        """Test that --version works."""
        from download_ravdess import __version__, main

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_main_shows_license(self, capsys: pytest.CaptureFixture) -> None:
        """Test that --license displays license notice."""
        from download_ravdess import main

        # This will try to download, but we just want to check license is shown
        # Use a non-existent output to trigger quick failure after license
        with patch("download_ravdess.download_from_zenodo") as mock_download:
            mock_download.return_value = 0
            main(["--license", "--quiet"])

        captured = capsys.readouterr()
        assert "CC BY-NC-SA 4.0" in captured.out or "Creative Commons" in captured.out


class TestExceptions:
    """Tests for custom exception classes."""

    def test_download_error(self) -> None:
        """Test DownloadError can be raised and caught."""
        with pytest.raises(DownloadError):
            raise DownloadError("Test error")

    def test_dependency_error(self) -> None:
        """Test DependencyError can be raised and caught."""
        with pytest.raises(DependencyError):
            raise DependencyError("Missing library")

    def test_exception_hierarchy(self) -> None:
        """Test exception class hierarchy."""
        from download_ravdess import (
            ChecksumError,
            ExtractionError,
            RAVDESSDownloadError,
        )

        assert issubclass(DownloadError, RAVDESSDownloadError)
        assert issubclass(ChecksumError, RAVDESSDownloadError)
        assert issubclass(ExtractionError, RAVDESSDownloadError)
        assert issubclass(DependencyError, RAVDESSDownloadError)
