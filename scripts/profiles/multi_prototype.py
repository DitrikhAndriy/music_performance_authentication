from pathlib import Path

import numpy as np

from scripts.storage.albums import list_albums
from scripts.storage.tracks import get_track_dir, list_tracks
from scripts.storage.utils import ARTISTS_DIR, now_iso, write_json


def collect_reference_tracks(artist_id: str) -> list[dict]:
    tracks = []

    for track in list_tracks(artist_id=artist_id):
        if track.get("is_reference", False):
            track = dict(track)
            track["album_id"] = None
            tracks.append(track)

    for album in list_albums(artist_id):
        album_id = album["album_id"]

        for track in list_tracks(artist_id=artist_id, album_id=album_id):
            if track.get("is_reference", False):
                track = dict(track)
                track["album_id"] = album_id
                tracks.append(track)

    return tracks


def load_track_embedding(
    artist_id: str,
    track: dict,
    feature_key: str
) -> np.ndarray | None:
    relative_path = track.get("features", {}).get(feature_key)

    if not relative_path:
        return None

    track_dir = get_track_dir(
        artist_id=artist_id,
        album_id=track.get("album_id"),
        track_id=track["track_id"]
    )
    path = track_dir / relative_path

    return np.load(path) if path.is_file() else None


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize_vector(a), normalize_vector(b)))


def initialize_centers(
    embeddings: np.ndarray,
    num_prototypes: int
) -> np.ndarray:
    if len(embeddings) <= num_prototypes:
        return embeddings.copy()

    indices = np.linspace(
        0,
        len(embeddings) - 1,
        num=num_prototypes,
        dtype=int
    )
    return embeddings[indices].copy()


def assign_to_centers(
    embeddings: np.ndarray,
    centers: np.ndarray
) -> list[int]:
    return [
        int(np.argmax([
            cosine_similarity(embedding, center)
            for center in centers
        ]))
        for embedding in embeddings
    ]


def recompute_centers(
    embeddings: np.ndarray,
    assignments: list[int],
    previous_centers: np.ndarray
) -> np.ndarray:
    new_centers = []
    assignments = np.asarray(assignments)

    for index, previous_center in enumerate(previous_centers):
        assigned = embeddings[assignments == index]
        center = (
            normalize_vector(np.mean(assigned, axis=0))
            if len(assigned)
            else previous_center
        )
        new_centers.append(center)

    return np.asarray(new_centers, dtype=np.float32)


def cluster_embeddings(
    embeddings: np.ndarray,
    num_prototypes: int,
    max_iterations: int = 20
) -> tuple[np.ndarray, list[int]]:
    if len(embeddings) == 0:
        return np.empty((0,), dtype=np.float32), []

    num_prototypes = min(num_prototypes, len(embeddings))
    embeddings = np.asarray([
        normalize_vector(embedding)
        for embedding in embeddings
    ], dtype=np.float32)

    centers = initialize_centers(embeddings, num_prototypes)
    previous_assignments = None

    for _ in range(max_iterations):
        assignments = assign_to_centers(embeddings, centers)

        if assignments == previous_assignments:
            break

        centers = recompute_centers(
            embeddings,
            assignments,
            centers
        )
        previous_assignments = assignments

    return centers, assignments


def build_multi_prototype_profile(
    artist_id: str,
    num_prototypes: int = 3
) -> dict:
    if num_prototypes <= 0:
        raise ValueError("num_prototypes must be greater than 0.")

    profile_dir = ARTISTS_DIR / artist_id / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    reference_tracks = collect_reference_tracks(artist_id)
    usable_tracks = []
    spectral_embeddings = []

    for track in reference_tracks:
        embedding = load_track_embedding(
            artist_id,
            track,
            "spectral_embedding_path"
        )

        if embedding is None:
            continue

        usable_tracks.append(track)
        spectral_embeddings.append(
            np.asarray(embedding, dtype=np.float32).reshape(-1)
        )

    if not spectral_embeddings:
        raise ValueError(
            f"No spectral embeddings found for artist {artist_id}."
        )

    centers, assignments = cluster_embeddings(
        np.stack(spectral_embeddings),
        num_prototypes
    )
    prototypes = []

    for index, center in enumerate(centers):
        prototype_id = f"P{index + 1:03d}"
        assigned_indices = [
            track_index
            for track_index, assignment in enumerate(assignments)
            if assignment == index
        ]

        prototype_tracks = []
        vocal_embeddings = []

        for track_index in assigned_indices:
            track = usable_tracks[track_index]
            vocal_embedding = load_track_embedding(
                artist_id,
                track,
                "vocal_embedding_path"
            )

            if vocal_embedding is not None:
                vocal_embeddings.append(
                    np.asarray(
                        vocal_embedding,
                        dtype=np.float32
                    ).reshape(-1)
                )

            prototype_tracks.append({
                "track_id": track["track_id"],
                "album_id": track.get("album_id"),
                "title": track.get("title"),
                "release_type": track.get("release_type"),
                "recording_type": track.get("recording_type")
            })

        spectral_filename = f"{prototype_id}_spectral_center.npy"
        np.save(
            profile_dir / spectral_filename,
            np.asarray(center, dtype=np.float32)
        )

        vocal_center_path = None

        if vocal_embeddings:
            vocal_center = normalize_vector(
                np.mean(np.stack(vocal_embeddings), axis=0)
            ).astype(np.float32)

            vocal_filename = f"{prototype_id}_vocal_center.npy"
            np.save(profile_dir / vocal_filename, vocal_center)
            vocal_center_path = f"profile/{vocal_filename}"

        prototypes.append({
            "prototype_id": prototype_id,
            "tracks_count": len(prototype_tracks),
            "tracks": prototype_tracks,
            "spectral_center_path": f"profile/{spectral_filename}",
            "vocal_center_path": vocal_center_path
        })

    timestamp = now_iso()
    profile = {
        "artist_id": artist_id,
        "profile_type": "multi_prototype",
        "reference_tracks_count": len(reference_tracks),
        "usable_tracks_count": len(usable_tracks),
        "requested_num_prototypes": num_prototypes,
        "actual_num_prototypes": len(prototypes),
        "clustering_feature": "spectral_embedding",
        "prototypes": prototypes,
        "created_at": timestamp,
        "updated_at": timestamp
    }

    write_json(
        profile_dir / "multi_prototype_profile.json",
        profile
    )
    return profile