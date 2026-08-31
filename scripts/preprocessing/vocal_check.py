import numpy as np
import soundfile as sf

from scripts.storage.tracks import get_track_dir
from scripts.storage.utils import read_json, write_json, now_iso


def check_vocals(
    artist_id: str,
    track_id: str,
    album_id: str | None = None,
    energy_threshold: float = 0.005
) -> dict:
    track_dir = get_track_dir(
        artist_id=artist_id,
        track_id=track_id,
        album_id=album_id
    )

    track_json = track_dir / "track.json"
    track = read_json(track_json)

    vocals_path = track["processed"].get("vocals_path")

    if not vocals_path:
        raise ValueError(
            "Vocals path is missing. Run separate_vocals() first."
        )

    full_vocals_path = track_dir / vocals_path

    if not full_vocals_path.exists():
        raise FileNotFoundError(
            f"Vocals file not found: {full_vocals_path}"
        )

    audio, _ = sf.read(full_vocals_path)

    if audio.size == 0:
        raise ValueError(f"Vocals file is empty: {full_vocals_path}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    rms_energy = float(np.sqrt(np.mean(np.square(audio))))
    has_vocals = rms_energy >= energy_threshold

    track["vocal_analysis"] = {
        "has_vocals": has_vocals,
        "rms_energy": rms_energy
    }

    track["updated_at"] = now_iso()

    write_json(track_json, track)

    return track