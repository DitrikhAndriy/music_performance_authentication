import numpy as np


def cosine_similarity(
    first_vector: np.ndarray,
    second_vector: np.ndarray
) -> float:
    first_vector = np.asarray(first_vector).reshape(-1)
    second_vector = np.asarray(second_vector).reshape(-1)

    if first_vector.shape != second_vector.shape:
        raise ValueError(
            "Vector dimensions do not match: "
            f"{first_vector.shape} and {second_vector.shape}"
        )

    denominator = (
        np.linalg.norm(first_vector)
        * np.linalg.norm(second_vector)
    )

    if denominator == 0:
        return 0.0

    score = np.dot(first_vector, second_vector) / denominator

    return float(np.clip(score, -1.0, 1.0))


def normalize_cosine(score: float) -> float:
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))