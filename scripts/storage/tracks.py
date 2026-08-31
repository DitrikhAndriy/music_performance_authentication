from pathlib import Path
import shutil
from scripts.storage.utils import (
    ARTISTS_DIR,
    ensure_dir,
    write_json,
    read_json,
    now_iso,
    generate_track_id,
    copy_file,
)


def get_track_dir(
    artist_id: str,
    track_id: str,
    album_id: str | None = None
) -> Path:
    artist_dir = ARTISTS_DIR / artist_id

    if album_id is None:
        return artist_dir / "tracks" / track_id

    return artist_dir / "albums" / album_id / "tracks" / track_id


def create_track(
    artist_id: str,
    title: str,
    audio_path: str,
    album_id: str | None = None,
    recording_type: str = "original",
    is_reference: bool = True,
    label: str = "authentic",
    cover_path: str | None = None
) -> dict:
    artist_dir = ARTISTS_DIR / artist_id

    if not artist_dir.exists():
        raise FileNotFoundError(f"Artist not found: {artist_id}")

    track_id = generate_track_id(artist_id)

    if album_id is None:
        release_type = "single"
        track_dir = artist_dir / "tracks" / track_id
    else:
        release_type = "album"
        album_dir = artist_dir / "albums" / album_id

        if not album_dir.exists():
            raise FileNotFoundError(f"Album not found: {album_id}")

        track_dir = album_dir / "tracks" / track_id

    ensure_dir(track_dir)
    ensure_dir(track_dir / "processed")
    ensure_dir(track_dir / "features")

    audio_ext = Path(audio_path).suffix.lower()
    audio_filename = f"original{audio_ext}"
    copy_file(audio_path, track_dir / audio_filename)

    cover_filename = None
    if cover_path:
        cover_filename = copy_file(cover_path, track_dir / "cover.jpg")

    track = {
        "track_id": track_id,
        "artist_id": artist_id,
        "album_id": album_id,
        "title": title,
        "release_type": release_type,
        "recording_type": recording_type,
        "is_reference": is_reference,
        "label": label,
        "audio_path": audio_filename,
        "cover_path": cover_filename,
        "processed": {
            "wav_path": None,
            "vocals_path": None
        },
        "features": {
            "vocal_embedding_path": None,
            "spectral_embedding_path": None,
            "ai_features_path": None
        },
        "created_at": now_iso(),
        "updated_at": now_iso()
    }

    write_json(track_dir / "track.json", track)
    return track


def get_track(
    artist_id: str,
    track_id: str,
    album_id: str | None = None
) -> dict:
    return read_json(get_track_dir(artist_id, track_id, album_id) / "track.json")


def list_tracks(
    artist_id: str,
    album_id: str | None = None,
    reference_only: bool = False,
    recording_type: str | None = None
) -> list[dict]:
    if album_id:
        tracks_dir = ARTISTS_DIR / artist_id / "albums" / album_id / "tracks"
    else:
        tracks_dir = ARTISTS_DIR / artist_id / "tracks"

    tracks = []

    if not tracks_dir.exists():
        return tracks

    for track_dir in tracks_dir.iterdir():
        track_json = track_dir / "track.json"
        if track_json.exists():
            track = read_json(track_json)

            if reference_only and not track.get("is_reference", False):
                continue

            if recording_type and track.get("recording_type") != recording_type:
                continue

            tracks.append(track)

    return tracks


def update_track(
    artist_id: str,
    track_id: str,
    album_id: str | None = None,
    title: str | None = None,
    recording_type: str | None = None,
    is_reference: bool | None = None,
    label: str | None = None,
    cover_path: str | None = None
) -> dict:
    track_dir = get_track_dir(artist_id, track_id, album_id)
    track_json = track_dir / "track.json"

    track = read_json(track_json)

    if title is not None:
        track["title"] = title

    if recording_type is not None:
        track["recording_type"] = recording_type

    if is_reference is not None:
        track["is_reference"] = is_reference

    if label is not None:
        track["label"] = label

    if cover_path is not None:
        track["cover_path"] = copy_file(cover_path, track_dir / "cover.jpg")

    track["updated_at"] = now_iso()

    write_json(track_json, track)
    return track


def delete_track(
    artist_id: str,
    track_id: str,
    album_id: str | None = None
) -> None:
    track_dir = get_track_dir(artist_id, track_id, album_id)

    if not track_dir.exists():
        raise FileNotFoundError(f"Track not found: {track_id}")

    shutil.rmtree(track_dir)