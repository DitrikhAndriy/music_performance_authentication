from pathlib import Path
import subprocess

from scripts.config import FFMPEG_EXE, check_ffmpeg
from scripts.storage.tracks import get_track_dir
from scripts.storage.utils import read_json, write_json, now_iso


def convert_to_wav(
    artist_id: str,
    track_id: str,
    album_id: str | None = None,
    sample_rate: int = 44100
) -> dict:
    check_ffmpeg()

    track_dir = get_track_dir(
        artist_id=artist_id,
        track_id=track_id,
        album_id=album_id
    )

    track_json = track_dir / "track.json"
    track = read_json(track_json)

    input_audio = track_dir / track["audio_path"]
    output_audio = track_dir / "processed" / "original.wav"

    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    output_audio.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(FFMPEG_EXE),
        "-y",
        "-i", str(input_audio),
        "-ac", "2",
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        str(output_audio)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr}")

    track["processed"]["wav_path"] = "processed/original.wav"
    track["updated_at"] = now_iso()

    write_json(track_json, track)

    return track