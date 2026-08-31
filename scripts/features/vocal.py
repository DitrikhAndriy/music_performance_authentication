import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from scripts.config import PROJECT_ROOT
from scripts.storage.tracks import get_track_dir
from scripts.storage.utils import read_json, write_json, now_iso


MODEL_DIR = PROJECT_ROOT / "tools" / "models" / "speechbrain" / "ecapa"

_CLASSIFIER = None
_CLASSIFIER_DEVICE = None


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_classifier(device: str) -> EncoderClassifier:
    global _CLASSIFIER
    global _CLASSIFIER_DEVICE

    if _CLASSIFIER is None or _CLASSIFIER_DEVICE != device:
        _CLASSIFIER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(MODEL_DIR),
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY
        )

        _CLASSIFIER_DEVICE = device

    return _CLASSIFIER


def extract_vocal_embedding(
    artist_id: str,
    track_id: str,
    album_id: str | None = None,
    sample_rate: int = 16000,
    skip_if_no_vocals: bool = True
) -> dict:
    track_dir = get_track_dir(
        artist_id=artist_id,
        track_id=track_id,
        album_id=album_id
    )

    track_json = track_dir / "track.json"
    track = read_json(track_json)

    vocal_analysis = track.get("vocal_analysis")

    if (
        skip_if_no_vocals
        and vocal_analysis
        and vocal_analysis.get("has_vocals") is False
    ):
        track["features"]["vocal_embedding_path"] = None
        track["features"]["vocal_embedding_dim"] = None
        track["features"]["vocal_embedding_model"] = None
        track["features"]["vocal_embedding_status"] = (
            "skipped_no_vocals"
        )

        track["updated_at"] = now_iso()
        write_json(track_json, track)

        return track

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

    waveform, current_sample_rate = torchaudio.load(
        full_vocals_path
    )

    if waveform.numel() == 0:
        raise ValueError(
            f"Vocals file is empty: {full_vocals_path}"
        )

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if current_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(
            current_sample_rate,
            sample_rate
        )

        waveform = resampler(waveform)

    device = get_device()
    classifier = get_classifier(device)

    waveform = waveform.to(device)

    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)

    embedding = (
        embedding
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    features_dir = track_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    output_path = features_dir / "vocal.npy"
    np.save(output_path, embedding)

    track["features"]["vocal_embedding_path"] = "features/vocal.npy"
    track["features"]["vocal_embedding_dim"] = int(embedding.shape[0])
    track["features"]["vocal_embedding_model"] = (
        "speechbrain_ecapa_tdnn"
    )
    track["features"]["vocal_embedding_status"] = "created"
    track["updated_at"] = now_iso()

    write_json(track_json, track)

    return track