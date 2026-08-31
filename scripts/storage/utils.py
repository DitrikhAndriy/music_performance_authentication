from pathlib import Path
import json
import shutil
from datetime import datetime

STORAGE_DIR = Path("storage")
ARTISTS_DIR = STORAGE_DIR / "artists"
INDEX_FILE = STORAGE_DIR / "artists" / "index.json"


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def init_storage() -> None:
    ensure_dir(ARTISTS_DIR)

    if not INDEX_FILE.exists():
        write_json(INDEX_FILE, {"next_artist_id": 1})


def generate_artist_id() -> str:
    init_storage()
    index = read_json(INDEX_FILE)

    number = index["next_artist_id"]
    index["next_artist_id"] += 1

    write_json(INDEX_FILE, index)
    return f"ART{number:03d}"


def generate_album_id(artist_id: str) -> str:
    artist_index = ARTISTS_DIR / artist_id / "index.json"
    index = read_json(artist_index)

    number = index["next_album_id"]
    index["next_album_id"] += 1

    write_json(artist_index, index)
    return f"ALB{number:03d}"


def generate_track_id(artist_id: str) -> str:
    artist_index = ARTISTS_DIR / artist_id / "index.json"
    index = read_json(artist_index)

    number = index["next_track_id"]
    index["next_track_id"] += 1

    write_json(artist_index, index)
    return f"TRK{number:03d}"


def copy_file(src: str | Path, dst: Path) -> str:
    src = Path(src)

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

    return dst.name