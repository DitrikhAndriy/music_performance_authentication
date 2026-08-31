from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys
import uuid
import numpy as np

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.features.spectral import extract_spectral_features
from scripts.preprocessing.convert import convert_to_wav
from scripts.preprocessing.probe import probe_audio
from scripts.preprocessing.vocal_check import check_vocals
from scripts.preprocessing.vocals import separate_vocals
from scripts.preprocessing.segments import create_segments
from scripts.similarity.spectral_similarity import load_spectral_features
from scripts.similarity.utils import cosine_similarity
from scripts.similarity.vocal_similarity import load_vocal_embedding
from scripts.profiles.multi_prototype import build_multi_prototype_profile
from scripts.storage.albums import create_album, list_albums
from scripts.storage.artists import create_artist, list_artists
from scripts.storage.tracks import create_track, get_track_dir, list_tracks
from scripts.storage.utils import ARTISTS_DIR, now_iso, read_json, write_json
from scripts.verification.svm_verification import verify_track


app = Flask(__name__)
app.secret_key = "local-masterwork-demo-secret"

UPLOADS_DIR = PROJECT_ROOT / "storage" / "raw" / "uploads"
MODELS_DIR = PROJECT_ROOT / "storage" / "models"
FINAL_CONFIG_PATH = MODELS_DIR / "final_svm_verification_config.json"
HISTORY_PATH = PROJECT_ROOT / "storage" / "verification_history.json"

ALLOWED_AUDIO = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
ALLOWED_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def load_json_if_exists(path: str | Path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def absolute_path(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def storage_artists_dir() -> Path:
    return absolute_path(ARTISTS_DIR)


def artist_dir(artist_id: str) -> Path:
    return storage_artists_dir() / artist_id


def album_dir(artist_id: str, album_id: str) -> Path:
    return artist_dir(artist_id) / "albums" / album_id


def track_dir(artist_id: str, track_id: str, album_id: str | None = None) -> Path:
    return get_track_dir(artist_id=artist_id, album_id=album_id, track_id=track_id)


def to_media_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    path = absolute_path(path)
    if not path.is_file():
        return None
    try:
        relative_path = path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return None
    return url_for("media", filename=relative_path)


def allowed_suffix(filename: str, allowed: set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed


def save_temp_upload(uploaded_file, allowed: set[str]) -> Path:
    if uploaded_file is None or not uploaded_file.filename:
        raise ValueError("File is required.")

    filename = secure_filename(uploaded_file.filename)
    suffix = Path(filename).suffix.lower()

    if not suffix and filename.lower() in {
        extension.lstrip(".") for extension in allowed
    }:
        suffix = f".{filename.lower()}"
        filename = f"audio{suffix}"

    if suffix not in allowed:
        raise ValueError(f"Unsupported file format: {suffix or filename}")

    output_path = UPLOADS_DIR / f"{uuid.uuid4().hex}_{filename}"
    uploaded_file.save(output_path)
    return output_path


def save_cover(uploaded_file, target_dir: Path) -> str | None:
    if uploaded_file is None or not uploaded_file.filename:
        return None
    filename = secure_filename(uploaded_file.filename)
    if not allowed_suffix(filename, ALLOWED_IMAGES):
        raise ValueError("Cover must be JPG, PNG or WEBP.")
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    cover_name = f"cover{suffix}"
    uploaded_file.save(target_dir / cover_name)
    return cover_name


def update_json_field(json_path: Path, **fields) -> None:
    data = read_json(json_path)
    data.update(fields)
    data["updated_at"] = now_iso()
    write_json(json_path, data)


def clean_title_from_filename(filename: str) -> str:
    name = Path(filename).stem.replace("_", " ").strip()
    # Removes common track-number prefixes like "01 - " or "01. ".
    parts = name.split(" - ", maxsplit=1)
    if len(parts) == 2 and parts[0].strip().isdigit():
        return parts[1].strip()
    if len(name) > 3 and name[:2].isdigit() and name[2] in {".", "-"}:
        return name[3:].strip()
    return name


def process_track(artist_id: str, track_id: str, album_id: str | None = None) -> None:
    convert_to_wav(artist_id=artist_id, album_id=album_id, track_id=track_id)
    probe_audio(artist_id=artist_id, album_id=album_id, track_id=track_id)
    separate_vocals(artist_id=artist_id, album_id=album_id, track_id=track_id)
    check_vocals(artist_id=artist_id, album_id=album_id, track_id=track_id)
    create_segments(artist_id=artist_id, album_id=album_id, track_id=track_id)
    extract_spectral_features(artist_id=artist_id, album_id=album_id, track_id=track_id)

    from scripts.features.vocal import extract_vocal_embedding
    extract_vocal_embedding(artist_id=artist_id, album_id=album_id, track_id=track_id)


def has_features(artist_id: str, track_id: str, album_id: str | None = None) -> bool:
    path = track_dir(artist_id, track_id, album_id)
    return (
        (path / "features" / "spectral.npy").is_file()
        and (path / "processed" / "vocals.wav").is_file()
    )


def history_items() -> list[dict]:
    return load_json_if_exists(HISTORY_PATH, default=[])


def save_history(items: list[dict]) -> None:
    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(items, file, ensure_ascii=False, indent=2)


def append_history(run: dict) -> None:
    items = history_items()
    items.insert(0, run)
    save_history(items[:200])


def model_config() -> dict | None:
    return load_json_if_exists(FINAL_CONFIG_PATH, default=None)


def result_score(result: dict) -> float:
    return float(
        result.get("authenticity_score")
        or result.get("score")
        or result.get("max_score")
        or 0.0
    )


def normalize_result(raw: dict, target_artist: dict, query_track: dict) -> dict:
    score = result_score(raw)
    threshold = raw.get("threshold")
    if threshold is None:
        config = model_config() or {}
        threshold = config.get("selected_threshold", 0.5)
    threshold = float(threshold)

    decision = raw.get("decision")
    if decision not in {"authentic", "not_authentic", "negative"}:
        decision = "authentic" if score >= threshold else "not_authentic"

    prototype_scores = raw.get("prototype_scores") or raw.get("prototypes") or []
    best_prototype_id = raw.get("best_prototype_id") or raw.get("best_prototype") or "—"

    return {
        "target_artist_id": target_artist["artist_id"],
        "target_artist_name": target_artist.get("name", target_artist["artist_id"]),
        "query_artist_id": query_track["artist_id"],
        "query_artist_name": query_track.get("artist_name", query_track["artist_id"]),
        "query_track_id": query_track["track_id"],
        "query_track_title": query_track.get("title", query_track["track_id"]),
        "query_album_id": query_track.get("album_id"),
        "score": score,
        "score_percent": round(score * 100, 2),
        "threshold": threshold,
        "decision": "authentic" if decision == "authentic" else "not_authentic",
        "best_prototype_id": best_prototype_id,
        "prototype_scores": prototype_scores,
        "raw": raw,
    }

def verification_similarities(raw: dict, target_artist_id: str, query_track: dict) -> dict:
    result = {"spectral_similarity": None, "vocal_similarity": None}
    prototype_id = raw.get("best_prototype_id")
    if not prototype_id:
        return result

    profile = read_json(
        artist_dir(target_artist_id)
        / "profile"
        / "multi_prototype_profile.json"
    )
    prototype = next(
        (
            item for item in profile.get("prototypes", [])
            if item.get("prototype_id") == prototype_id
        ),
        None
    )
    if prototype is None:
        return result

    query_spectral = load_spectral_features(
        artist_id=query_track["artist_id"],
        track_id=query_track["track_id"],
        album_id=query_track.get("album_id")
    )
    prototype_spectral = np.load(
        artist_dir(target_artist_id)
        / prototype["spectral_center_path"]
    ).astype(np.float32).reshape(-1)

    result["spectral_similarity"] = cosine_similarity(
        query_spectral,
        prototype_spectral
    )

    vocal_path = prototype.get("vocal_center_path")
    if not vocal_path:
        return result

    try:
        query_vocal = load_vocal_embedding(
            artist_id=query_track["artist_id"],
            track_id=query_track["track_id"],
            album_id=query_track.get("album_id")
        )
    except (ValueError, FileNotFoundError):
        return result

    full_vocal_path = artist_dir(target_artist_id) / vocal_path
    if not full_vocal_path.is_file():
        return result

    prototype_vocal = np.load(
        full_vocal_path
    ).astype(np.float32).reshape(-1)

    result["vocal_similarity"] = cosine_similarity(
        query_vocal,
        prototype_vocal
    )
    return result

# -----------------------------------------------------------------------------
# Data for page
# -----------------------------------------------------------------------------

def track_json_path(artist_id: str, track_id: str, album_id: str | None = None) -> Path:
    return track_dir(artist_id, track_id, album_id) / "track.json"


def enrich_track(track: dict, artist: dict, album: dict | None = None) -> dict:
    item = dict(track)
    item["artist_id"] = artist["artist_id"]
    item["artist_name"] = artist.get("name", artist["artist_id"])
    item["album_id"] = album["album_id"] if album else None
    item["album_title"] = album.get("title") if album else "Single"
    item["processed"] = has_features(item["artist_id"], item["track_id"], item["album_id"])

    cover_path = item.get("cover_path")
    item["cover_url"] = (
        to_media_url(track_dir(item["artist_id"], item["track_id"], item["album_id"]) / cover_path)
        if cover_path else None
    )
    return item


def library_data() -> tuple[list[dict], list[dict]]:
    artists_result = []
    all_tracks = []

    for artist in list_artists():
        artist = dict(artist)
        artist_id = artist["artist_id"]
        a_dir = artist_dir(artist_id)
        artist["cover_url"] = to_media_url(a_dir / artist.get("cover_path")) if artist.get("cover_path") else None
        artist["profile_ready"] = (a_dir / "profile" / "multi_prototype_profile.json").is_file()

        singles = [enrich_track(track, artist) for track in list_tracks(artist_id=artist_id)]
        albums = []

        for album in list_albums(artist_id):
            album = dict(album)
            album_id = album["album_id"]
            alb_dir = album_dir(artist_id, album_id)
            album["cover_url"] = to_media_url(alb_dir / album.get("cover_path")) if album.get("cover_path") else None
            album["tracks"] = [
                enrich_track(track, artist, album)
                for track in list_tracks(artist_id=artist_id, album_id=album_id)
            ]
            albums.append(album)
            all_tracks.extend(album["tracks"])

        artist["singles"] = singles
        artist["albums"] = albums
        artist["tracks_count"] = len(singles) + sum(len(album["tracks"]) for album in albums)
        artist["reference_tracks_count"] = sum(1 for t in singles if t.get("is_reference")) + sum(
            1 for album in albums for t in album["tracks"] if t.get("is_reference")
        )
        artists_result.append(artist)
        all_tracks.extend(singles)

    return artists_result, all_tracks


def track_choices(all_tracks: list[dict]) -> list[dict]:
    choices = []
    for track in all_tracks:
        choices.append({
            "value": f"{track['artist_id']}|{track.get('album_id') or 'None'}|{track['track_id']}",
            "label": f"{track['artist_name']} — {track.get('title', track['track_id'])}",
        })
    return choices


def dashboard_stats(artists: list[dict], all_tracks: list[dict]) -> dict:
    config = load_json_if_exists(FINAL_CONFIG_PATH)
    model_path = absolute_path(config["model_path"]) if config and config.get("model_path") else None

    return {
        "artists_count": len(artists),
        "tracks_count": len(all_tracks),
        "profiles_count": sum(artist["profile_ready"] for artist in artists),
        "history_count": len(history_items()),
        "threshold": config.get("selected_threshold") if config else None,
        "model_ready": bool(FINAL_CONFIG_PATH.is_file() and model_path and model_path.is_file()),
    }


def page_data(extra: dict | None = None) -> dict:
    artists, all_tracks = library_data()
    data = {
        "artists": artists,
        "all_tracks": all_tracks,
        "track_choices": track_choices(all_tracks),
        "stats": dashboard_stats(artists, all_tracks),
        "history": history_items(),
        "verification_results": None,
    }
    if extra:
        data.update(extra)
    return data


def find_artist(artist_id: str) -> dict | None:
    for artist in list_artists():
        if artist["artist_id"] == artist_id:
            return dict(artist)
    return None


def find_track(artist_id: str, track_id: str, album_id: str | None = None) -> dict | None:
    for artist in list_artists():
        if artist["artist_id"] != artist_id:
            continue
        album = None
        if album_id:
            album = next((dict(a) for a in list_albums(artist_id) if a["album_id"] == album_id), None)
        for track in list_tracks(artist_id=artist_id, album_id=album_id):
            if track["track_id"] == track_id:
                return enrich_track(track, dict(artist), album)
    return None


# -----------------------------------------------------------------------------
# Routes and actions
# -----------------------------------------------------------------------------

@app.route("/media/<path:filename>")
def media(filename):
    return send_from_directory(PROJECT_ROOT.resolve(), filename)


@app.route("/", methods=["GET", "POST"])
def index():
    extra = {}
    if request.method == "POST":
        action = request.form.get("action")
        try:
            if action == "create_artist":
                handle_create_artist()
                flash("Artist created.", "success")
                return redirect(url_for("index") + "#library")

            if action == "delete_artist":
                artist_id = require_form("artist_id")
                shutil.rmtree(artist_dir(artist_id), ignore_errors=True)
                flash("Artist deleted.", "success")
                return redirect(url_for("index") + "#library")

            if action == "update_cover":
                handle_update_cover()
                flash("Cover updated.", "success")
                return redirect(url_for("index") + "#library")

            if action == "create_album":
                handle_create_album()
                flash("Album created.", "success")
                return redirect(url_for("index") + "#library")

            if action == "create_album_with_tracks":
                handle_create_album_with_tracks()
                flash("Album and tracks created.", "success")
                return redirect(url_for("index") + "#library")

            if action == "delete_album":
                artist_id = require_form("artist_id")
                album_id = require_form("album_id")
                shutil.rmtree(album_dir(artist_id, album_id), ignore_errors=True)
                flash("Album deleted.", "success")
                return redirect(url_for("index") + "#library")

            if action == "add_single":
                handle_add_single()
                flash("Single added.", "success")
                return redirect(url_for("index") + "#library")

            if action == "delete_track":
                artist_id = require_form("artist_id")
                track_id = require_form("track_id")
                album_id = nullable_form("album_id")
                shutil.rmtree(track_dir(artist_id, track_id, album_id), ignore_errors=True)
                flash("Track deleted.", "success")
                return redirect(url_for("index") + "#library")

            if action == "process_track":
                artist_id = require_form("artist_id")
                track_id = require_form("track_id")
                album_id = nullable_form("album_id")
                process_track(artist_id, track_id, album_id)
                flash("Track processed.", "success")
                return redirect(url_for("index") + "#library")

            if action == "process_album":
                artist_id = require_form("artist_id")
                album_id = require_form("album_id")
                for track in list_tracks(artist_id=artist_id, album_id=album_id):
                    process_track(artist_id, track["track_id"], album_id)
                flash("Album tracks processed.", "success")
                return redirect(url_for("index") + "#library")

            if action == "build_profile":
                artist_id = require_form("profile_artist_id")
                build_multi_prototype_profile(artist_id=artist_id, num_prototypes=3)
                flash("Artist profile rebuilt.", "success")
                return redirect(url_for("index") + "#library")

            if action == "verify_existing":
                extra["verification_results"] = handle_verify_existing()
                flash("Verification completed and saved to history.", "success")

            if action == "clear_history":
                save_history([])
                flash("History cleared.", "success")
                return redirect(url_for("index"))

            if action == "toggle_track_reference":
                artist_id = require_form("artist_id")
                album_id = request.form.get("album_id") or None
                track_id = require_form("track_id")

                path = track_json_path(artist_id, track_id, album_id)
                track = read_json(path)
                is_reference = not track.get("is_reference", False)

                update_json_field(
                    path,
                    is_reference=is_reference,
                    label="authentic" if is_reference else "unknown"
                )

                return redirect(url_for("index") + "#library")

        except Exception as error:
            flash(str(error), "error")

    return render_template("index.html", **page_data(extra))


def require_form(name: str) -> str:
    value = request.form.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing field: {name}")
    return value


def nullable_form(name: str) -> str | None:
    value = request.form.get(name, "").strip()
    return None if value in {"", "None", "null"} else value


def checkbox(name: str) -> bool:
    return request.form.get(name) == "on"


def handle_create_artist() -> None:
    name = require_form("artist_name")
    artist = create_artist(name)
    cover_name = save_cover(request.files.get("artist_cover"), artist_dir(artist["artist_id"]))
    if cover_name:
        update_json_field(artist_dir(artist["artist_id"]) / "artist.json", cover_path=cover_name)


def handle_update_cover() -> None:
    entity = require_form("entity")
    artist_id = require_form("artist_id")
    album_id = nullable_form("album_id")
    track_id = nullable_form("track_id")

    if entity == "artist":
        target_dir = artist_dir(artist_id)
        json_path = target_dir / "artist.json"
    elif entity == "album":
        if not album_id:
            raise ValueError("Album id is required.")
        target_dir = album_dir(artist_id, album_id)
        json_path = target_dir / "album.json"
    elif entity == "track":
        if not track_id:
            raise ValueError("Track id is required.")
        target_dir = track_dir(artist_id, track_id, album_id)
        json_path = target_dir / "track.json"
    else:
        raise ValueError("Unknown cover entity.")

    cover_name = save_cover(request.files.get("cover_file"), target_dir)
    if not cover_name:
        raise ValueError("Cover file is required.")
    update_json_field(json_path, cover_path=cover_name)


def handle_create_album() -> None:
    artist_id = require_form("album_artist_id")
    title = require_form("album_title")
    album = create_album(artist_id=artist_id, title=title)
    cover_name = save_cover(request.files.get("album_cover"), album_dir(artist_id, album["album_id"]))
    if cover_name:
        update_json_field(album_dir(artist_id, album["album_id"]) / "album.json", cover_path=cover_name)


def create_track_from_upload(
    artist_id: str,
    audio_file,
    title: str,
    album_id: str | None = None,
    is_reference: bool = True,
    process_now: bool = False,
):
    temp_path = save_temp_upload(audio_file, ALLOWED_AUDIO)
    try:
        track = create_track(
            artist_id=artist_id,
            album_id=album_id,
            title=title,
            audio_path=str(temp_path),
            recording_type="original",
            is_reference=is_reference,
            label="authentic" if is_reference else "unknown",
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if process_now:
        process_track(artist_id, track["track_id"], album_id)
    return track


def handle_create_album_with_tracks() -> None:
    artist_id = require_form("album_artist_id")
    title = require_form("album_title")

    album = next((item for item in list_albums(artist_id) if item.get("title", "").strip().casefold() == title.casefold()), None)

    if album is None:
        album = create_album(artist_id=artist_id,title=title)

    album_id = album["album_id"]

    cover_name = save_cover(request.files.get("album_cover"), album_dir(artist_id, album_id))
    if cover_name:
        update_json_field(album_dir(artist_id, album_id) / "album.json", cover_path=cover_name)

    files = [file for file in request.files.getlist("album_tracks") if file and file.filename]
    titles = request.form.getlist("album_track_titles")
    reference_indices = set(request.form.getlist("album_track_reference"))
    process_now = checkbox("process_album_after_upload")

    for index, audio_file in enumerate(files):
        track_title = titles[index].strip() if index < len(titles) and titles[index].strip() else clean_title_from_filename(audio_file.filename)
        is_reference = str(index) in reference_indices
        create_track_from_upload(artist_id, audio_file, track_title, album_id, is_reference, process_now=False)

    if process_now:
        for track in list_tracks(artist_id=artist_id, album_id=album_id):
            process_track(artist_id, track["track_id"], album_id)


def handle_add_single() -> None:
    artist_id = require_form("single_artist_id")
    audio_file = request.files.get("single_audio")
    title = request.form.get("single_title", "").strip()
    if audio_file is None or not audio_file.filename:
        raise ValueError("Audio file is required.")
    title = title or clean_title_from_filename(audio_file.filename)
    track = create_track_from_upload(
        artist_id=artist_id,
        audio_file=audio_file,
        title=title,
        album_id=None,
        is_reference=checkbox("single_is_reference"),
        process_now=True,
    )
    cover_name = save_cover(request.files.get("single_cover"), track_dir(artist_id, track["track_id"]))
    if cover_name:
        update_json_field(track_json_path(artist_id, track["track_id"]), cover_path=cover_name)


def handle_verify_existing() -> list[dict]:
    selected_track = require_form("query_track")
    query_artist_id, query_album_id, query_track_id = selected_track.split("|", maxsplit=2)
    query_album_id = None if query_album_id == "None" else query_album_id
    target_artist_ids = request.form.getlist("target_artist_ids")

    if not target_artist_ids:
        raise ValueError("Select at least one target artist.")

    query_track = find_track(query_artist_id, query_track_id, query_album_id)
    if not query_track:
        raise ValueError("Selected track was not found.")

    if not has_features(query_artist_id, query_track_id, query_album_id):
        process_track(query_artist_id, query_track_id, query_album_id)

    results = []

    for target_artist_id in target_artist_ids:
        target_artist = find_artist(target_artist_id)
        if not target_artist:
            continue

        raw = verify_track(
            target_artist_id=target_artist_id,
            track_artist_id=query_artist_id,
            track_id=query_track_id,
            album_id=query_album_id,
            profile_type="multi_prototype",
        )

        result = normalize_result(raw, target_artist, query_track)
        result.update(
            verification_similarities(
                raw=raw,
                target_artist_id=target_artist_id,
                query_track=query_track,
            )
        )
        results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)

    append_history({
        "history_id": uuid.uuid4().hex,
        "created_at": now_iso(),
        "query_track": {
            "artist_id": query_artist_id,
            "artist_name": query_track.get("artist_name"),
            "album_id": query_album_id,
            "track_id": query_track_id,
            "title": query_track.get("title"),
        },
        "results": results,
    })

    return results


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )