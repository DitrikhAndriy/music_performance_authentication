from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from scripts.datasets.multi_prototype_loader import (
    normalize_vector,
    prepare_artifact_features,
    prepare_feature_vector
)
from scripts.storage.artists import get_artist
from scripts.storage.tracks import get_track_dir
from scripts.storage.utils import ARTISTS_DIR, read_json, write_json


CONFIG_PATH = Path("storage/models/final_svm_verification_config.json")
HISTORY_PATH = Path("storage/history/verification_history.json")


def load_npy(path: Path | None) -> np.ndarray | None:
    return np.load(path) if path and path.is_file() else None


@lru_cache(maxsize=2)
def load_model(model_path: str):
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(f"SVM model not found: {path}")

    return joblib.load(path)


@lru_cache(maxsize=32)
def load_profile(artist_id: str, profile_type: str) -> dict:
    profile_dir = ARTISTS_DIR / artist_id / "profile"

    paths = {
        "multi_prototype":
            profile_dir / "multi_prototype_profile.json",
        "holdout_multi_prototype":
            profile_dir / "holdout" / "holdout_multi_prototype_profile.json",
        "verification_multi_prototype":
            profile_dir / "verification" / "verification_multi_prototype_profile.json"
    }

    if profile_type not in paths:
        raise ValueError(f"Unknown profile type: {profile_type}")

    path = paths[profile_type]

    if not path.is_file():
        raise FileNotFoundError(f"Profile not found: {path}")

    return read_json(path)


def load_track(
    artist_id: str,
    track_id: str,
    album_id: str | None
) -> tuple[dict, Path]:
    track_dir = get_track_dir(artist_id, track_id, album_id)
    path = track_dir / "track.json"

    if not path.is_file():
        raise FileNotFoundError(f"Track metadata not found: {path}")

    return read_json(path), track_dir


def track_feature_path(
    track: dict,
    track_dir: Path,
    key: str
) -> Path | None:
    path = track.get("features", {}).get(key)
    return track_dir / path if path else None


def prototype_feature_path(
    artist_id: str,
    prototype: dict,
    key: str
) -> Path | None:
    path = prototype.get(key)
    return ARTISTS_DIR / artist_id / path if path else None


def cosine_similarity(
    first: np.ndarray | None,
    second: np.ndarray | None
) -> float | None:
    if first is None or second is None:
        return None

    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)

    if first.size != second.size:
        return None

    denominator = np.linalg.norm(first) * np.linalg.norm(second)

    if denominator == 0:
        return None

    return float(np.dot(first, second) / denominator)


def build_vector(
    track: dict,
    track_dir: Path,
    target_artist_id: str,
    prototype: dict,
    spectral_dim: int,
    vocal_dim: int,
    include_artifacts: bool
) -> tuple[np.ndarray, dict]:
    track_spectral_raw = load_npy(
        track_feature_path(track, track_dir, "spectral_embedding_path")
    )
    track_vocal_raw = load_npy(
        track_feature_path(track, track_dir, "vocal_embedding_path")
    )
    prototype_spectral_raw = load_npy(
        prototype_feature_path(
            target_artist_id,
            prototype,
            "spectral_center_path"
        )
    )
    prototype_vocal_raw = load_npy(
        prototype_feature_path(
            target_artist_id,
            prototype,
            "vocal_center_path"
        )
    )

    has_track_vocal = (
        track_vocal_raw is not None
        and np.asarray(track_vocal_raw).size == vocal_dim
    )
    has_prototype_vocal = (
        prototype_vocal_raw is not None
        and np.asarray(prototype_vocal_raw).size == vocal_dim
    )

    track_spectral = normalize_vector(
        prepare_feature_vector(track_spectral_raw, spectral_dim, required=True)
    )
    prototype_spectral = normalize_vector(
        prepare_feature_vector(
            prototype_spectral_raw,
            spectral_dim,
            required=True
        )
    )
    track_vocal = normalize_vector(
        prepare_feature_vector(track_vocal_raw, vocal_dim)
    )
    prototype_vocal = normalize_vector(
        prepare_feature_vector(prototype_vocal_raw, vocal_dim)
    )

    parts = [
        track_spectral,
        prototype_spectral,
        np.abs(track_spectral - prototype_spectral),
        track_vocal,
        prototype_vocal,
        np.abs(track_vocal - prototype_vocal),
        np.array(
            [float(has_track_vocal), float(has_prototype_vocal)],
            dtype=np.float32
        )
    ]

    if include_artifacts:
        artifact_raw = load_npy(
            track_feature_path(track, track_dir, "artifact_features_path")
        )
        artifact_features, has_artifacts = prepare_artifact_features(
            artifact_raw
        )
        parts.extend([
            artifact_features,
            np.array([has_artifacts], dtype=np.float32)
        ])

    details = {
        "spectral_similarity": cosine_similarity(
            track_spectral_raw,
            prototype_spectral_raw
        ),
        "vocal_similarity": (
            cosine_similarity(track_vocal_raw, prototype_vocal_raw)
            if has_track_vocal and has_prototype_vocal
            else None
        ),
        "has_track_vocal": has_track_vocal,
        "has_prototype_vocal": has_prototype_vocal
    }

    return np.concatenate(parts).astype(np.float32), details


def predict_score(model, vector: np.ndarray) -> float:
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Positive class 1 is absent in the SVM model.")

    probabilities = model.predict_proba(vector.reshape(1, -1))
    return float(probabilities[0, classes.index(1)])


def verify_track(
    target_artist_id: str,
    track_artist_id: str,
    track_id: str,
    album_id: str | None = None,
    config_path: str | Path = CONFIG_PATH,
    profile_type: str | None = None
) -> dict:
    config = read_json(Path(config_path))
    model = load_model(str(config["model_path"]))

    threshold = float(config["selected_threshold"])
    spectral_dim = int(config["spectral_dim"])
    vocal_dim = int(config["vocal_dim"])
    include_artifacts = bool(
        config.get("artifact_features_used_as_model_input", False)
    )
    profile_type = profile_type or config.get(
        "profile_type_for_inference",
        "multi_prototype"
    )

    track, track_dir = load_track(
        track_artist_id,
        track_id,
        album_id
    )
    profile = load_profile(target_artist_id, profile_type)
    prototypes = profile.get("prototypes", [])

    if not prototypes:
        raise ValueError(f"Profile {target_artist_id} has no prototypes.")

    prototype_results = []

    for prototype in prototypes:
        vector, details = build_vector(
            track,
            track_dir,
            target_artist_id,
            prototype,
            spectral_dim,
            vocal_dim,
            include_artifacts
        )

        input_dim = int(
            config.get(
                "input_dim",
                getattr(model, "n_features_in_", vector.size)
            )
        )

        if vector.size != input_dim:
            raise ValueError(
                f"Feature vector dimension is {vector.size}, "
                f"but model expects {input_dim}."
            )

        prototype_results.append({
            "prototype_id": prototype["prototype_id"],
            "svm_score": predict_score(model, vector),
            **details
        })

    best = max(
        prototype_results,
        key=lambda item: item["svm_score"]
    )
    score = float(best["svm_score"])

    return {
        "target_artist_id": target_artist_id,
        "target_artist_name": get_artist(target_artist_id).get("name"),
        "track_artist_id": track_artist_id,
        "album_id": album_id,
        "track_id": track_id,
        "track_title": track.get("title"),
        "profile_type": profile_type,
        "authenticity_score": score,
        "threshold": threshold,
        "decision": "authentic" if score >= threshold else "not_authentic",
        "best_prototype_id": best["prototype_id"],
        "spectral_similarity": best["spectral_similarity"],
        "vocal_similarity": best["vocal_similarity"],
        "has_track_vocal": best["has_track_vocal"],
        "has_prototype_vocal": best["has_prototype_vocal"],
        "prototype_results": prototype_results
    }


def save_history(result: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = read_json(HISTORY_PATH) if HISTORY_PATH.is_file() else []
    history.append(result)
    write_json(HISTORY_PATH, history)


def verify_track_against_artists(
    target_artist_ids: list[str],
    track_artist_id: str,
    track_id: str,
    album_id: str | None = None,
    profile_type: str | None = None,
    save_result: bool = True
) -> dict:
    if not target_artist_ids:
        raise ValueError("At least one target artist must be selected.")

    results = [
        verify_track(
            target_artist_id=target_artist_id,
            track_artist_id=track_artist_id,
            track_id=track_id,
            album_id=album_id,
            profile_type=profile_type
        )
        for target_artist_id in dict.fromkeys(target_artist_ids)
    ]

    verification = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "track": {
            "artist_id": track_artist_id,
            "album_id": album_id,
            "track_id": track_id,
            "title": results[0]["track_title"]
        },
        "selected_artists_count": len(results),
        "results": [
            {
                "target_artist_id": item["target_artist_id"],
                "target_artist_name": item["target_artist_name"],
                "authenticity_score": item["authenticity_score"],
                "threshold": item["threshold"],
                "decision": item["decision"],
                "best_prototype_id": item["best_prototype_id"],
                "spectral_similarity": item["spectral_similarity"],
                "vocal_similarity": item["vocal_similarity"]
            }
            for item in results
        ]
    }

    if save_result:
        save_history(verification)

    return verification