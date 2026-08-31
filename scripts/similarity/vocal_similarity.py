import numpy as np

from scripts.similarity.utils import cosine_similarity
from scripts.storage.tracks import get_track_dir
from scripts.storage.utils import read_json


def load_vocal_embedding(
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

    vocal_path = (
        track
        .get("features", {})
        .get("vocal_embedding_path")
    )

    if not vocal_path:
        raise ValueError(
            "Vocal embedding path is missing. "
            "Run extract_vocal_embedding() first."
        )

    full_vocal_path = track_dir / vocal_path

    if not full_vocal_path.exists():
        raise FileNotFoundError(
            f"Vocal embedding not found: {full_vocal_path}"
        )

    embedding = np.load(full_vocal_path)

    if embedding.size == 0:
        raise ValueError(
            f"Vocal embedding is empty: {full_vocal_path}"
        )

    return embedding.astype(np.float32).reshape(-1)


def compare_vocal_embeddings(
    artist_id_a: str,
    track_id_a: str,
    artist_id_b: str,
    track_id_b: str,
    album_id_a: str | None = None,
    album_id_b: str | None = None
) -> float:
    first_embedding = load_vocal_embedding(
        artist_id=artist_id_a,
        track_id=track_id_a,
        album_id=album_id_a
    )

    second_embedding = load_vocal_embedding(
        artist_id=artist_id_b,
        track_id=track_id_b,
        album_id=album_id_b
    )

    return cosine_similarity(
        first_embedding,
        second_embedding
    )