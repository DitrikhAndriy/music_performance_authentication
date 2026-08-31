import librosa
import numpy as np

from scripts.storage.tracks import get_track_dir
from scripts.storage.utils import read_json, write_json, now_iso


def extract_spectral_features(
    artist_id: str,
    track_id: str,
    album_id: str | None = None
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

    full_wav_path = track_dir / wav_path

    if not full_wav_path.exists():
        raise FileNotFoundError(
            f"WAV file not found: {full_wav_path}"
        )

    audio, sample_rate = librosa.load(
        full_wav_path,
        sr=None,
        mono=True
    )

    if audio.size == 0:
        raise ValueError(f"WAV file is empty: {full_wav_path}")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=20
    )

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate
    )

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sample_rate
    )

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate
    )

    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

    feature_vector = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        chroma.mean(axis=1),
        chroma.std(axis=1),
        contrast.mean(axis=1),
        contrast.std(axis=1),
        centroid.mean(axis=1),
        centroid.std(axis=1),
        bandwidth.mean(axis=1),
        bandwidth.std(axis=1),
        rolloff.mean(axis=1),
        rolloff.std(axis=1),
        zero_crossing_rate.mean(axis=1),
        zero_crossing_rate.std(axis=1)
    ]).astype(np.float32)

    features_dir = track_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    output_path = features_dir / "spectral.npy"
    np.save(output_path, feature_vector)

    track["features"]["spectral_embedding_path"] = (
        "features/spectral.npy"
    )
    track["features"]["spectral_embedding_dim"] = int(
        feature_vector.shape[0]
    )

    track["updated_at"] = now_iso()

    write_json(track_json, track)

    return track