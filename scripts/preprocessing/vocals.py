import os
import shutil
import subprocess
import sys

from scripts.config import FFMPEG_EXE
from scripts.storage.tracks import get_track_dir
from scripts.storage.utils import read_json, write_json, now_iso


def separate_vocals(
    artist_id: str,
    track_id: str,
    album_id: str | None = None,
    model_name: str = "htdemucs"
) -> dict:
    track_dir = get_track_dir(
        artist_id=artist_id,
        track_id=track_id,
        album_id=album_id
    )

    track_json = track_dir / "track.json"
    track = read_json(track_json)

    wav_path = track["processed"].get("wav_path")

    if not wav_path:
        raise ValueError(
            "WAV path is missing. Run convert_to_wav() first."
        )

    input_wav = track_dir / wav_path

    if not input_wav.exists():
        raise FileNotFoundError(
            f"Processed WAV not found: {input_wav}"
        )

    processed_dir = track_dir / "processed"
    demucs_output_dir = processed_dir / "demucs_temp"

    command = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        model_name,
        "--two-stems",
        "vocals",
        "--out",
        str(demucs_output_dir),
        str(input_wav)
    ]

    env = os.environ.copy()
    env["PATH"] = (
        str(FFMPEG_EXE.parent)
        + os.pathsep
        + env.get("PATH", "")
    )

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Demucs error:\n{result.stderr}"
            )

        source_vocals = (
            demucs_output_dir
            / model_name
            / input_wav.stem
            / "vocals.wav"
        )

        if not source_vocals.exists():
            raise FileNotFoundError(
                f"Demucs vocals output not found: {source_vocals}"
            )

        target_vocals = processed_dir / "vocals.wav"
        shutil.copy2(source_vocals, target_vocals)

    finally:
        if demucs_output_dir.exists():
            shutil.rmtree(demucs_output_dir)

    track["processed"]["vocals_path"] = "processed/vocals.wav"
    track["updated_at"] = now_iso()

    write_json(track_json, track)

    return track