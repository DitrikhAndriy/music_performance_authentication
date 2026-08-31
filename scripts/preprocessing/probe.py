import json
import subprocess

from scripts.config import FFPROBE_EXE, check_ffmpeg
from scripts.storage.tracks import get_track_dir
from scripts.storage.utils import read_json, write_json, now_iso


def probe_audio(
    artist_id: str,
    track_id: str,
    album_id: str | None = None
) -> dict:
    check_ffmpeg()

    track_dir = get_track_dir(
        artist_id=artist_id,
        track_id=track_id,
        album_id=album_id
    )

    track_json = track_dir / "track.json"
    track = read_json(track_json)

    audio_path = track_dir / track["audio_path"]

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    command = [
        str(FFPROBE_EXE),
        "-v", "error",
        "-show_entries",
        "format=duration,bit_rate:stream=sample_rate,channels",
        "-of", "json",
        str(audio_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFprobe error:\n{result.stderr}")

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid FFprobe JSON output:\n{result.stdout}"
        ) from error

    format_info = info.get("format", {})
    streams = info.get("streams", [])

    stream_info = streams[0] if streams else {}

    track["audio_metadata"] = {
        "duration": float(format_info.get("duration") or 0),
        "bitrate": int(format_info.get("bit_rate") or 0),
        "sample_rate": int(stream_info.get("sample_rate") or 0),
        "channels": int(stream_info.get("channels") or 0)
    }

    track["updated_at"] = now_iso()

    write_json(track_json, track)

    return track