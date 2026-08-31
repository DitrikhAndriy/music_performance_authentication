from pathlib import Path
import shutil

import soundfile as sf

from scripts.storage.tracks import get_track_dir
from scripts.storage.utils import read_json, write_json, now_iso


def split_audio_file(
    input_path: Path,
    output_dir: Path,
    segment_seconds: int = 20
) -> list[str]:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be greater than 0.")

    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = sf.read(input_path)

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {input_path}")

    segment_samples = segment_seconds * sample_rate
    total_samples = len(audio)

    saved_segments = []

    for start in range(0, total_samples, segment_samples):
        end = min(start + segment_samples, total_samples)
        segment = audio[start:end]

        if len(segment) < segment_samples * 0.5:
            continue

        segment_index = len(saved_segments) + 1
        segment_name = f"segment_{segment_index:03d}.wav"
        segment_path = output_dir / segment_name

        sf.write(segment_path, segment, sample_rate)

        relative_path = segment_path.relative_to(
            output_dir.parent.parent
        )

        saved_segments.append(str(relative_path))

    return saved_segments


def create_segments(
    artist_id: str,
    track_id: str,
    album_id: str | None = None,
    segment_seconds: int = 20
) -> dict:
    track_dir = get_track_dir(
        artist_id=artist_id,
        track_id=track_id,
        album_id=album_id
    )

    track_json = track_dir / "track.json"
    track = read_json(track_json)

    processed = track["processed"]

    wav_path = processed.get("wav_path")
    vocals_path = processed.get("vocals_path")

    if not wav_path:
        raise ValueError(
            "WAV path is missing. Run convert_to_wav() first."
        )

    original_segments = split_audio_file(
        input_path=track_dir / wav_path,
        output_dir=track_dir / "processed" / "segments" / "original",
        segment_seconds=segment_seconds
    )

    vocal_segments = []

    if vocals_path:
        vocal_segments = split_audio_file(
            input_path=track_dir / vocals_path,
            output_dir=track_dir / "processed" / "segments" / "vocals",
            segment_seconds=segment_seconds
        )

    track["processed"]["segments"] = {
        "segment_seconds": segment_seconds,
        "original": original_segments,
        "vocals": vocal_segments
    }

    track["updated_at"] = now_iso()

    write_json(track_json, track)

    return track