"""Resumable downloader for the official MuCo-3DHP source annotations.

MuCo-3DHP is generated locally from MPI-INF-3DHP.  For trajectory models, the
useful compact source files are ``annot.mat`` and ``camera.calibration`` for all
8 subjects and 2 sequences.  The RGB videos and masks are intentionally not
downloaded by this script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


BASE_URL = "https://vcai.mpi-inf.mpg.de/3dhp-dataset"
CHUNK_BYTES = 8 * 1024 * 1024
USER_AGENT = "Mozilla/5.0 (compatible; MuCo-3DHP research downloader)"


def request(url: str, *, byte_range: tuple[int, int] | None = None):
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return urllib.request.Request(url, headers=headers)


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download_part(job: tuple[str, Path, int, int]) -> tuple[Path, int]:
    url, part_path, start, end = job
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        return part_path, expected

    part_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = part_path.with_suffix(".tmp")
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(
                request(url, byte_range=(start, end)), timeout=120
            ) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(
                    f"bytes {start}-{end}/"
                ):
                    raise RuntimeError(
                        f"server ignored byte range {start}-{end}: "
                        f"status={response.status}, Content-Range={content_range!r}"
                    )
                with temp_path.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            if temp_path.stat().st_size != expected:
                raise RuntimeError(
                    f"short part {part_path.name}: "
                    f"{temp_path.stat().st_size} != {expected}"
                )
            os.replace(temp_path, part_path)
            return part_path, expected
        except (OSError, RuntimeError, urllib.error.URLError):
            if attempt == 5:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def assemble(destination: Path, size: int, parts: list[Path]) -> None:
    temp_path = destination.with_suffix(destination.suffix + ".download")
    with temp_path.open("wb") as output:
        for part_path in parts:
            with part_path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
    if temp_path.stat().st_size != size:
        raise RuntimeError(
            f"assembled size mismatch for {destination}: "
            f"{temp_path.stat().st_size} != {size}"
        )
    os.replace(temp_path, destination)
    for part_path in parts:
        part_path.unlink()
    parts[0].parent.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    files: list[tuple[str, Path, int, list[Path]]] = []
    jobs: list[tuple[str, Path, int, int]] = []
    total_bytes = 0

    for subject in range(1, 9):
        for sequence in range(1, 3):
            for filename in ("annot.mat", "camera.calibration"):
                relative = Path(f"S{subject}") / f"Seq{sequence}" / filename
                url = f"{BASE_URL}/{relative.as_posix()}"
                destination = args.target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                size = remote_size(url)
                total_bytes += size
                if destination.exists() and destination.stat().st_size == size:
                    print(f"verified existing: {relative} ({size:,} bytes)", flush=True)
                    continue

                part_dir = destination.parent / f".{filename}.parts"
                part_paths: list[Path] = []
                for start in range(0, size, CHUNK_BYTES):
                    end = min(start + CHUNK_BYTES, size) - 1
                    part_path = part_dir / f"{start:012d}-{end:012d}.part"
                    part_paths.append(part_path)
                    jobs.append((url, part_path, start, end))
                files.append((url, destination, size, part_paths))

    print(
        f"downloading {len(jobs)} parts with {args.workers} workers; "
        f"dataset total={total_bytes / 1024**3:.2f} GiB",
        flush=True,
    )
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_part, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            _, count = future.result()
            completed += count
            if completed // (64 * 1024 * 1024) != (
                completed - count
            ) // (64 * 1024 * 1024):
                print(
                    f"downloaded/resumed {completed / 1024**2:.0f} MiB",
                    flush=True,
                )

    for _, destination, size, parts in files:
        assemble(destination, size, parts)
        print(
            f"assembled: {destination.relative_to(args.target)} "
            f"({size:,} bytes)",
            flush=True,
        )
    print("DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
