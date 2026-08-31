import numpy as np

from scripts.similarity.utils import cosine_similarity
from scripts.storage.tracks import get_track_dir
from scripts.storage.utils import read_json


def load_spectral_features(
    artist_id: str,
    track_id: str,
    album_id: str | None = None
) -> np.ndarray:
    track_dir = get_track_dir(
        artist_id=artist_id,
        track_id=track_id,
        album_id=album_id
    )

    track = read_json(track_dir / "track.json")

    spectral_path = (
        track
        .get("features", {})
        .get("spectral_embedding_path")
    )

    if not spectral_path:
        raise ValueError(
            "Spectral feature path is missing. "
            "Run extract_spectral_features() first."
        )

    full_spectral_path = track_dir / spectral_path

    if not full_spectral_path.exists():
        raise FileNotFoundError(
            f"Spectral features not found: {full_spectral_path}"
        )

    features = np.load(full_spectral_path)

    if features.size == 0:
        raise ValueError(
            f"Spectral feature vector is empty: {full_spectral_path}"
        )

    return features.astype(np.float32).reshape(-1)


def compare_spectral_features(
    artist_id_a: str,
    track_id_a: str,
    artist_id_b: str,
    track_id_b: str,
    album_id_a: str | None = None,
    album_id_b: str | None = None
) -> float:
    first_features = load_spectral_features(
        artist_id=artist_id_a,
        track_id=track_id_a,
        album_id=album_id_a
    )

    second_features = load_spectral_features(
        artist_id=artist_id_b,
        track_id=track_id_b,
        album_id=album_id_b
    )

    return cosine_similarity(
        first_features,
        second_features
    )