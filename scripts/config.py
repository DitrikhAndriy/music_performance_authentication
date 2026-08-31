from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOOLS_DIR = PROJECT_ROOT / "tools"
FFMPEG_EXE = TOOLS_DIR / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE_EXE = TOOLS_DIR / "ffmpeg" / "bin" / "ffprobe.exe"


def check_ffmpeg() -> None:
    if not FFMPEG_EXE.exists():
        raise FileNotFoundError(
            f"FFmpeg not found: {FFMPEG_EXE}\n"
            "Download FFmpeg and place it into tools/ffmpeg/"
        )

    if not FFPROBE_EXE.exists():
        raise FileNotFoundError(
            f"FFprobe not found: {FFPROBE_EXE}\n"
            "Download FFmpeg and place it into tools/ffmpeg/"
        )