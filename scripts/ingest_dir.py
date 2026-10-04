"""Bulk-load a folder into the brain without the web UI: python -m scripts.ingest_dir /path/to/folder [--source-type email]"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import uuid
from pathlib import Path

from psycopg.types.json import Jsonb

from mybrain.config import get_settings
from mybrain.db import connection, run_migrations
from mybrain.ingest import queue as jobq
from mybrain.ingest.parsers import TEXT_EXTS

EXTS = TEXT_EXTS | {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xlsm",
    ".csv",
    ".tsv",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".eml",
    ".mbox",
    ".html",
    ".htm",
    ".json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--source-type", default="upload")
    ap.add_argument("--tag", action="append", default=[])
    args = ap.parse_args()
    run_migrations()
    s = get_settings()
    root = Path(args.folder)
    added = skipped = 0
    with connection() as conn:
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS):
            data = path.read_bytes()
            h = hashlib.sha256(data).hexdigest()
            if conn.execute("SELECT 1 FROM documents WHERE content_hash = %s", (h,)).fetchone():
                skipped += 1
                continue
            doc_id = str(uuid.uuid4())
            dest = s.uploads_dir / doc_id / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            conn.execute(
                "INSERT INTO documents (id, content_hash, filename, source_type, bytes, storage_path, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    doc_id,
                    h,
                    path.name,
                    args.source_type,
                    len(data),
                    str(dest),
                    Jsonb({"tags": args.tag, "origin": str(path.relative_to(root))}),
                ),
            )
            conn.commit()
            jobq.enqueue(conn, doc_id, "ingest")
            added += 1
    print(f"queued {added} documents, skipped {skipped} duplicates")


if __name__ == "__main__":
    main()
