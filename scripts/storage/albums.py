import shutil

from scripts.storage.utils import (
    ARTISTS_DIR,
    ensure_dir,
    write_json,
    read_json,
    now_iso,
    generate_album_id,
    copy_file,
)


def create_album(
    artist_id: str,
    title: str,
    cover_path: str | None = None
) -> dict:
    artist_dir = ARTISTS_DIR / artist_id

    if not artist_dir.exists():
        raise FileNotFoundError(f"Artist not found: {artist_id}")

    album_id = generate_album_id(artist_id)
    album_dir = artist_dir / "albums" / album_id

    ensure_dir(album_dir)
    ensure_dir(album_dir / "tracks")

    cover_filename = None
    if cover_path:
        cover_filename = copy_file(cover_path, album_dir / "cover.jpg")

    album = {
        "album_id": album_id,
        "artist_id": artist_id,
        "title": title,
        "cover_path": cover_filename,
        "created_at": now_iso(),
        "updated_at": now_iso()
    }

    write_json(album_dir / "album.json", album)
    return album


def get_album(artist_id: str, album_id: str) -> dict:
    return read_json(
        ARTISTS_DIR / artist_id / "albums" / album_id / "album.json"
    )


def list_albums(artist_id: str) -> list[dict]:
    albums_dir = ARTISTS_DIR / artist_id / "albums"
    albums = []

    if not albums_dir.exists():
        return albums

    for album_dir in albums_dir.iterdir():
        album_json = album_dir / "album.json"

        if album_json.exists():
            albums.append(read_json(album_json))

    return albums


def update_album(
    artist_id: str,
    album_id: str,
    title: str | None = None,
    cover_path: str | None = None
) -> dict:
    album_dir = ARTISTS_DIR / artist_id / "albums" / album_id
    album_json = album_dir / "album.json"

    album = read_json(album_json)

    if title is not None:
        album["title"] = title

    if cover_path is not None:
        album["cover_path"] = copy_file(
            cover_path,
            album_dir / "cover.jpg"
        )

    album["updated_at"] = now_iso()

    write_json(album_json, album)
    return album


def delete_album(artist_id: str, album_id: str) -> None:
    album_dir = ARTISTS_DIR / artist_id / "albums" / album_id

    if not album_dir.exists():
        raise FileNotFoundError(f"Album not found: {album_id}")

    shutil.rmtree(album_dir)