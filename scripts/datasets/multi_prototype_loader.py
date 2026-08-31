from pathlib import Path

import numpy as np

from scripts.storage.utils import ARTISTS_DIR, read_json, write_json


ARTIFACT_FEATURE_NAMES = [
    "normalized_peak_ratio",
    "periodicity_score",
    "normalized_peak_strength"
]
ARTIFACT_FEATURE_DIM = len(ARTIFACT_FEATURE_NAMES)

ARTIFACT_INPUT_NAMES = [
    *ARTIFACT_FEATURE_NAMES,
    "has_track_artifact_features"
]
ARTIFACT_INPUT_DIM = len(ARTIFACT_INPUT_NAMES)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def load_npy_if_exists(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    return np.load(path)


def resolve_track_feature_path(item: dict, feature_path_key: str) -> Path | None:
    relative_path = item.get(feature_path_key)
    if not relative_path:
        return None
    return Path(item["track_dir"]) / relative_path


def resolve_prototype_feature_path(
    item: dict,
    feature_path_key: str
) -> Path | None:
    relative_path = item.get(feature_path_key)
    if not relative_path:
        return None

    return ARTISTS_DIR / item["target_artist_id"] / relative_path


def prepare_feature_vector(
    vector: np.ndarray | None,
    expected_dim: int,
    required: bool = False
) -> np.ndarray:
    if vector is None:
        if required:
            raise ValueError("Required feature vector is missing.")
        return np.zeros(expected_dim, dtype=np.float32)

    vector = np.asarray(vector, dtype=np.float32).reshape(-1)

    if vector.size != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {expected_dim}, "
            f"got {vector.size}."
        )

    return vector


def has_valid_vector(
    vector: np.ndarray | None,
    expected_dim: int
) -> float:
    if vector is None:
        return 0.0

    return float(np.asarray(vector).reshape(-1).size == expected_dim)


def infer_dimension(
    items: list[dict],
    track_key: str,
    prototype_key: str
) -> int | None:
    for item in items:
        paths = [
            resolve_track_feature_path(item, track_key),
            resolve_prototype_feature_path(item, prototype_key)
        ]

        for path in paths:
            vector = load_npy_if_exists(path)

            if vector is not None:
                return int(np.asarray(vector).reshape(-1).size)

    return None


def infer_feature_dimensions(items: list[dict]) -> dict:
    spectral_dim = infer_dimension(
        items,
        "track_spectral_embedding_path",
        "prototype_spectral_center_path"
    )
    vocal_dim = infer_dimension(
        items,
        "track_vocal_embedding_path",
        "prototype_vocal_center_path"
    )

    if spectral_dim is None:
        raise ValueError("Could not infer spectral embedding dimension.")

    if vocal_dim is None:
        raise ValueError("Could not infer vocal embedding dimension.")

    return {
        "spectral_dim": spectral_dim,
        "vocal_dim": vocal_dim
    }


def prepare_artifact_features(
    artifact_vector: np.ndarray | None
) -> tuple[np.ndarray, float]:
    if artifact_vector is None:
        return np.zeros(ARTIFACT_FEATURE_DIM, dtype=np.float32), 0.0

    artifact_vector = np.asarray(
        artifact_vector,
        dtype=np.float32
    ).reshape(-1)

    if artifact_vector.size != 5:
        raise ValueError(
            "Artifact feature dimension mismatch: "
            f"expected 5, got {artifact_vector.size}."
        )

    features = np.array([
        np.clip(float(artifact_vector[1]) / 0.05, 0.0, 1.0),
        np.clip(float(artifact_vector[2]), 0.0, 1.0),
        np.clip(float(artifact_vector[3]) / 5.0, 0.0, 1.0)
    ], dtype=np.float32)

    return features, 1.0


def build_feature_vector(
    item: dict,
    spectral_dim: int,
    vocal_dim: int,
    include_artifact_features: bool
) -> np.ndarray:
    track_spectral_raw = load_npy_if_exists(
        resolve_track_feature_path(
            item,
            "track_spectral_embedding_path"
        )
    )
    prototype_spectral_raw = load_npy_if_exists(
        resolve_prototype_feature_path(
            item,
            "prototype_spectral_center_path"
        )
    )
    track_vocal_raw = load_npy_if_exists(
        resolve_track_feature_path(
            item,
            "track_vocal_embedding_path"
        )
    )
    prototype_vocal_raw = load_npy_if_exists(
        resolve_prototype_feature_path(
            item,
            "prototype_vocal_center_path"
        )
    )

    has_track_vocal = has_valid_vector(track_vocal_raw, vocal_dim)
    has_prototype_vocal = has_valid_vector(
        prototype_vocal_raw,
        vocal_dim
    )

    track_spectral = normalize_vector(
        prepare_feature_vector(
            track_spectral_raw,
            spectral_dim,
            required=True
        )
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

    feature_parts = [
        track_spectral,
        prototype_spectral,
        np.abs(track_spectral - prototype_spectral),
        track_vocal,
        prototype_vocal,
        np.abs(track_vocal - prototype_vocal),
        np.array(
            [has_track_vocal, has_prototype_vocal],
            dtype=np.float32
        )
    ]

    if include_artifact_features:
        artifact_raw = load_npy_if_exists(
            resolve_track_feature_path(
                item,
                "track_artifact_features_path"
            )
        )
        artifact_features, has_artifacts = prepare_artifact_features(
            artifact_raw
        )

        feature_parts.extend([
            artifact_features,
            np.array([has_artifacts], dtype=np.float32)
        ])

    return np.concatenate(feature_parts).astype(np.float32)


def build_siamese_feature_vectors(
    item: dict,
    spectral_dim: int,
    vocal_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track_spectral_raw = load_npy_if_exists(
        resolve_track_feature_path(
            item,
            "track_spectral_embedding_path"
        )
    )
    prototype_spectral_raw = load_npy_if_exists(
        resolve_prototype_feature_path(
            item,
            "prototype_spectral_center_path"
        )
    )
    track_vocal_raw = load_npy_if_exists(
        resolve_track_feature_path(
            item,
            "track_vocal_embedding_path"
        )
    )
    prototype_vocal_raw = load_npy_if_exists(
        resolve_prototype_feature_path(
            item,
            "prototype_vocal_center_path"
        )
    )

    has_track_vocal = has_valid_vector(
        track_vocal_raw,
        vocal_dim
    )
    has_prototype_vocal = has_valid_vector(
        prototype_vocal_raw,
        vocal_dim
    )

    track_spectral = normalize_vector(
        prepare_feature_vector(
            track_spectral_raw,
            spectral_dim,
            required=True
        )
    )
    prototype_spectral = normalize_vector(
        prepare_feature_vector(
            prototype_spectral_raw,
            spectral_dim,
            required=True
        )
    )
    track_vocal = normalize_vector(
        prepare_feature_vector(
            track_vocal_raw,
            vocal_dim
        )
    )
    prototype_vocal = normalize_vector(
        prepare_feature_vector(
            prototype_vocal_raw,
            vocal_dim
        )
    )

    track_features = np.concatenate([
        track_spectral,
        track_vocal
    ]).astype(np.float32)

    prototype_features = np.concatenate([
        prototype_spectral,
        prototype_vocal
    ]).astype(np.float32)

    extra_features = np.array([
        has_track_vocal,
        has_prototype_vocal
    ], dtype=np.float32)

    return track_features, prototype_features, extra_features


def build_siamese_arrays_from_items(
    items: list[dict],
    spectral_dim: int,
    vocal_dim: int
) -> dict:
    track_features = []
    prototype_features = []
    extra_features = []
    labels = []
    verification_ids = []
    metadata = []
    skipped_items = []

    for item in items:
        try:
            track, prototype, extra = (
                build_siamese_feature_vectors(
                    item,
                    spectral_dim,
                    vocal_dim
                )
            )

            track_features.append(track)
            prototype_features.append(prototype)
            extra_features.append(extra)
            labels.append(int(item["label"]))
            verification_ids.append(
                item.get("verification_id") or ""
            )

            metadata.append({
                "split": item.get("split"),
                "verification_id": item.get("verification_id"),
                "target_artist_id": item.get(
                    "target_artist_id"
                ),
                "track_artist_id": item.get(
                    "track_artist_id"
                ),
                "album_id": item.get("album_id"),
                "track_id": item.get("track_id"),
                "track_title": item.get("track_title"),
                "prototype_id": item.get("prototype_id"),
                "label": item.get("label"),
                "pair_type": item.get("pair_type")
            })
        except Exception as error:
            skipped_items.append({
                "item": item,
                "reason": str(error)
            })

    if not track_features:
        raise ValueError(
            "No valid Siamese items were converted into arrays."
        )

    return {
        "X_track": np.stack(
            track_features
        ).astype(np.float32),
        "X_prototype": np.stack(
            prototype_features
        ).astype(np.float32),
        "X_extra": np.stack(
            extra_features
        ).astype(np.float32),
        "y": np.asarray(
            labels,
            dtype=np.float32
        ),
        "verification_ids": np.asarray(
            verification_ids,
            dtype=str
        ),
        "metadata": metadata,
        "skipped_items": skipped_items
    }


def build_arrays_from_items(
    items: list[dict],
    spectral_dim: int,
    vocal_dim: int,
    include_artifact_features: bool
) -> dict:
    features = []
    labels = []
    verification_ids = []
    metadata = []
    skipped_items = []

    for item in items:
        try:
            features.append(
                build_feature_vector(
                    item,
                    spectral_dim,
                    vocal_dim,
                    include_artifact_features
                )
            )
            labels.append(int(item["label"]))
            verification_ids.append(item.get("verification_id") or "")

            metadata.append({
                "split": item.get("split"),
                "verification_id": item.get("verification_id"),
                "target_artist_id": item.get("target_artist_id"),
                "track_artist_id": item.get("track_artist_id"),
                "album_id": item.get("album_id"),
                "track_id": item.get("track_id"),
                "track_title": item.get("track_title"),
                "prototype_id": item.get("prototype_id"),
                "label": item.get("label"),
                "pair_type": item.get("pair_type"),
                "track_artifact_features_path": item.get(
                    "track_artifact_features_path"
                )
            })
        except Exception as error:
            skipped_items.append({
                "item": item,
                "reason": str(error)
            })

    if not features:
        raise ValueError("No valid items were converted into arrays.")

    return {
        "X": np.stack(features).astype(np.float32),
        "y": np.asarray(labels, dtype=np.float32),
        "verification_ids": np.asarray(verification_ids, dtype=str),
        "metadata": metadata,
        "skipped_items": skipped_items
    }


def build_input_structure(
    include_artifact_features: bool
) -> list[str]:
    structure = [
        "track_spectral",
        "prototype_spectral",
        "spectral_difference",
        "track_vocal",
        "prototype_vocal",
        "vocal_difference",
        "has_track_vocal",
        "has_prototype_vocal"
    ]

    if include_artifact_features:
        structure.extend(ARTIFACT_INPUT_NAMES)

    return structure


def build_split_summary(
    split_name: str,
    items: list[dict],
    arrays: dict
) -> dict:
    prefix = f"{split_name}_"

    summary = {
        f"{prefix}items_count": len(items),
        f"{prefix}used_items_count": len(arrays["X"]),
        f"{prefix}skipped_items_count": len(arrays["skipped_items"])
    }

    if split_name == "train":
        summary.update({
            "train_positive_count": int(np.sum(arrays["y"] == 1)),
            "train_negative_count": int(np.sum(arrays["y"] == 0))
        })
    else:
        summary.update({
            f"{prefix}positive_pair_count": int(
                np.sum(arrays["y"] == 1)
            ),
            f"{prefix}negative_pair_count": int(
                np.sum(arrays["y"] == 0)
            ),
            f"{prefix}verification_count": len(
                set(arrays["verification_ids"].tolist())
            )
        })

    return summary


def save_holdout_training_npz(
    dataset_path: str | Path = (
        "storage/datasets/holdout_multi_prototype_pairs.json"
    ),
    output_path: str | Path = (
        "storage/datasets/holdout_multi_prototype_training_data.npz"
    ),
    metadata_output_path: str | Path = (
        "storage/datasets/"
        "holdout_multi_prototype_training_metadata.json"
    ),
    include_artifact_features: bool = False
) -> dict:
    dataset_path = Path(dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = read_json(dataset_path)

    train_items = dataset.get("train_items", [])
    validation_items = dataset.get("validation_items", [])
    test_items = dataset.get("test_items", [])

    if not train_items:
        raise ValueError("Dataset contains no train items.")

    if not validation_items:
        raise ValueError("Dataset contains no validation items.")

    if not test_items:
        raise ValueError("Dataset contains no test items.")

    dimensions = infer_feature_dimensions(
        train_items + validation_items + test_items
    )
    spectral_dim = dimensions["spectral_dim"]
    vocal_dim = dimensions["vocal_dim"]

    train_arrays = build_arrays_from_items(
        train_items,
        spectral_dim,
        vocal_dim,
        include_artifact_features
    )
    validation_arrays = build_arrays_from_items(
        validation_items,
        spectral_dim,
        vocal_dim,
        include_artifact_features
    )
    test_arrays = build_arrays_from_items(
        test_items,
        spectral_dim,
        vocal_dim,
        include_artifact_features
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        X_train=train_arrays["X"],
        y_train=train_arrays["y"],
        X_validation=validation_arrays["X"],
        y_validation=validation_arrays["y"],
        validation_verification_ids=validation_arrays[
            "verification_ids"
        ],
        X_test=test_arrays["X"],
        y_test=test_arrays["y"],
        test_verification_ids=test_arrays["verification_ids"]
    )

    summary = {
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "metadata_output_path": str(metadata_output_path),
        "input_dim": int(train_arrays["X"].shape[1]),
        "spectral_dim": spectral_dim,
        "vocal_dim": vocal_dim,
        "artifact_input_dim": (
            ARTIFACT_INPUT_DIM if include_artifact_features else 0
        ),
        **build_split_summary("train", train_items, train_arrays),
        **build_split_summary(
            "validation",
            validation_items,
            validation_arrays
        ),
        **build_split_summary("test", test_items, test_arrays),
        "input_structure": build_input_structure(
            include_artifact_features
        ),
        "artifact_features_used_as_model_input": (
            include_artifact_features
        )
    }

    metadata = {
        "summary": summary,
        "train_metadata": train_arrays["metadata"],
        "validation_metadata": validation_arrays["metadata"],
        "test_metadata": test_arrays["metadata"],
        "train_skipped_items": train_arrays["skipped_items"],
        "validation_skipped_items": validation_arrays["skipped_items"],
        "test_skipped_items": test_arrays["skipped_items"]
    }

    write_json(Path(metadata_output_path), metadata)
    return summary


def save_holdout_siamese_npz(
    dataset_path: str | Path = (
        "storage/datasets/"
        "holdout_multi_prototype_pairs.json"
    ),
    output_path: str | Path = (
        "storage/datasets/"
        "holdout_multi_prototype_siamese_data.npz"
    ),
    metadata_output_path: str | Path = (
        "storage/datasets/"
        "holdout_multi_prototype_siamese_metadata.json"
    )
) -> dict:
    dataset_path = Path(dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    dataset = read_json(dataset_path)

    train_items = dataset.get("train_items", [])
    validation_items = dataset.get(
        "validation_items",
        []
    )
    test_items = dataset.get("test_items", [])

    if not train_items:
        raise ValueError(
            "Dataset contains no train items."
        )

    if not validation_items:
        raise ValueError(
            "Dataset contains no validation items."
        )

    if not test_items:
        raise ValueError(
            "Dataset contains no test items."
        )

    dimensions = infer_feature_dimensions(
        train_items
        + validation_items
        + test_items
    )

    spectral_dim = dimensions["spectral_dim"]
    vocal_dim = dimensions["vocal_dim"]

    train_arrays = build_siamese_arrays_from_items(
        train_items,
        spectral_dim,
        vocal_dim
    )
    validation_arrays = build_siamese_arrays_from_items(
        validation_items,
        spectral_dim,
        vocal_dim
    )
    test_arrays = build_siamese_arrays_from_items(
        test_items,
        spectral_dim,
        vocal_dim
    )

    output_path = Path(output_path)
    metadata_output_path = Path(
        metadata_output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        output_path,

        X_track_train=train_arrays["X_track"],
        X_prototype_train=train_arrays["X_prototype"],
        X_extra_train=train_arrays["X_extra"],
        y_train=train_arrays["y"],

        X_track_validation=validation_arrays["X_track"],
        X_prototype_validation=(
            validation_arrays["X_prototype"]
        ),
        X_extra_validation=validation_arrays["X_extra"],
        y_validation=validation_arrays["y"],
        validation_verification_ids=validation_arrays[
            "verification_ids"
        ],

        X_track_test=test_arrays["X_track"],
        X_prototype_test=test_arrays["X_prototype"],
        X_extra_test=test_arrays["X_extra"],
        y_test=test_arrays["y"],
        test_verification_ids=test_arrays[
            "verification_ids"
        ]
    )

    summary = {
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "metadata_output_path": str(
            metadata_output_path
        ),
        "spectral_dim": spectral_dim,
        "vocal_dim": vocal_dim,
        "branch_input_dim": (
            spectral_dim + vocal_dim
        ),
        "extra_input_dim": 2,
        "train_items_count": int(
            len(train_arrays["y"])
        ),
        "train_positive_count": int(
            np.sum(train_arrays["y"] == 1)
        ),
        "train_negative_count": int(
            np.sum(train_arrays["y"] == 0)
        ),
        "validation_items_count": int(
            len(validation_arrays["y"])
        ),
        "validation_positive_count": int(
            np.sum(validation_arrays["y"] == 1)
        ),
        "validation_negative_count": int(
            np.sum(validation_arrays["y"] == 0)
        ),
        "validation_verification_count": int(
            len(
                set(
                    validation_arrays[
                        "verification_ids"
                    ].tolist()
                )
            )
        ),
        "test_items_count": int(
            len(test_arrays["y"])
        ),
        "test_positive_count": int(
            np.sum(test_arrays["y"] == 1)
        ),
        "test_negative_count": int(
            np.sum(test_arrays["y"] == 0)
        ),
        "test_verification_count": int(
            len(
                set(
                    test_arrays[
                        "verification_ids"
                    ].tolist()
                )
            )
        ),
        "train_skipped_items_count": len(
            train_arrays["skipped_items"]
        ),
        "validation_skipped_items_count": len(
            validation_arrays["skipped_items"]
        ),
        "test_skipped_items_count": len(
            test_arrays["skipped_items"]
        ),
        "branch_input_structure": [
            "spectral_embedding",
            "vocal_embedding"
        ],
        "extra_input_structure": [
            "has_track_vocal",
            "has_prototype_vocal"
        ]
    }

    metadata = {
        "summary": summary,
        "train_metadata": train_arrays["metadata"],
        "validation_metadata": (
            validation_arrays["metadata"]
        ),
        "test_metadata": test_arrays["metadata"],
        "train_skipped_items": (
            train_arrays["skipped_items"]
        ),
        "validation_skipped_items": (
            validation_arrays["skipped_items"]
        ),
        "test_skipped_items": (
            test_arrays["skipped_items"]
        )
    }

    write_json(metadata_output_path, metadata)
    return summary