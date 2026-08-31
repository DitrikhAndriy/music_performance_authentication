import shutil
from scripts.storage.utils import (
    ARTISTS_DIR,
    ensure_dir,
    write_json,
    read_json,
    now_iso,
    generate_artist_id,
    copy_file,
)


def create_artist(name: str, cover_path: str | None = None) -> dict:
    artist_id = generate_artist_id()
    artist_dir = ARTISTS_DIR / artist_id

    ensure_dir(artist_dir)
    ensure_dir(artist_dir / "albums")
    ensure_dir(artist_dir / "tracks")

    write_json(artist_dir / "index.json", {
        "next_album_id": 1,
        "next_track_id": 1
    })

    cover_filename = None
    if cover_path:
        cover_filename = copy_file(cover_path, artist_dir / "cover.jpg")

    artist = {
        "artist_id": artist_id,
        "name": name,
        "cover_path": cover_filename,
        "created_at": now_iso(),
        "updated_at": now_iso()
    }

    write_json(artist_dir / "artist.json", artist)
    return artist


def get_artist(artist_id: str) -> dict:
    return read_json(ARTISTS_DIR / artist_id / "artist.json")


def list_artists() -> list[dict]:
    artists = []

    if not ARTISTS_DIR.exists():
        return artists

    for artist_dir in ARTISTS_DIR.iterdir():
        artist_json = artist_dir / "artist.json"
        if artist_json.exists():
            artists.append(read_json(artist_json))

    return artists


def update_artist(
    artist_id: str,
    name: str | None = None,
    cover_path: str | None = None
) -> dict:
    artist_dir = ARTISTS_DIR / artist_id
    artist_json = artist_dir / "artist.json"

    artist = read_json(artist_json)

    if name is not None:
        artist["name"] = name

    if cover_path is not None:
        artist["cover_path"] = copy_file(cover_path, artist_dir / "cover.jpg")

    artist["updated_at"] = now_iso()

    write_json(artist_json, artist)
    return artist


def delete_artist(artist_id: str) -> None:
    artist_dir = ARTISTS_DIR / artist_id

    if not artist_dir.exists():
        raise FileNotFoundError(f"Artist not found: {artist_id}")

    shutil.rmtree(artist_dir)