"""Download only the PHercParis4 spiral inputs this benchmark uses (~15.5 GB).

    python fetch_inputs.py DEST

Mirrors from https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/ with
48 threads; resumable (files already present are skipped). Data agreement:
https://dl.ash2txt.org/LICENSE.txt. Do not redistribute the downloaded files.
"""
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests

BASE = "https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/"
FILES = ["umbilicus.json", "abs_winding.json", "relative_windings.json", "same_windings.json"]
DIRS = ["outer_shell/", "verified_patches/", "eval_fibers/",
        "lasagna_inputs/las_008_nx.ome.zarr.respool_g4_pair/"]
HREF = re.compile(r'href="([^"?][^"]*)"')

session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))
lock = threading.Lock()
stats = {"files": 0, "bytes": 0}


def listing(url):
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return [h for h in HREF.findall(r.text) if not h.startswith(("/", "..", "http"))]


def fetch(rel, dest):
    path = dest / unquote(rel)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(urljoin(BASE, rel), timeout=120, stream=True) as r:
        r.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    with lock:
        stats["files"] += 1
        stats["bytes"] += path.stat().st_size
        if stats["files"] % 2000 == 0:
            print(f"{stats['files']} files {stats['bytes'] / 1e9:.2f} GB", flush=True)


def walk(rel, dest, pool, futures):
    for h in listing(urljoin(BASE, rel)):
        if h.endswith("/"):
            futures.append(pool.submit(walk, rel + h, dest, pool, futures))
        else:
            futures.append(pool.submit(fetch, rel + h, dest))


def main(dest):
    dest = Path(dest)
    with ThreadPoolExecutor(48) as pool:
        futures = [pool.submit(fetch, f, dest) for f in FILES]
        futures += [pool.submit(walk, d, dest, pool, futures) for d in DIRS]
        i = 0
        while i < len(futures):
            futures[i].result()
            i += 1
    print(f"DONE {stats['files']} new files {stats['bytes'] / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] in (["-h"], ["--help"]):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
