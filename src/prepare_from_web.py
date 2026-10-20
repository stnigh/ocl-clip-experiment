import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the remaining OCL web images with retries."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--proxy", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--attempts", type=int, default=3)
    return parser.parse_args()


def is_valid_image(path, min_size=32):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(str(path)) as image:
            image.verify()
        with Image.open(str(path)) as image:
            image.load()
            if image.width < min_size or image.height < min_size:
                return False
        return True
    except Exception:
        return False


def make_proxies(proxy):
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def download_one(
    url,
    relative_path,
    output_root,
    timeout,
    attempts,
    proxies,
):
    target = output_root / Path(relative_path)
    if is_valid_image(target):
        return "skipped", relative_path, ""

    target.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(target) + ".part")
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(
                url,
                headers=HEADERS,
                proxies=proxies,
                timeout=timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                with open(str(part), "wb") as fp:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            fp.write(chunk)
            if not is_valid_image(part):
                raise ValueError("Downloaded file is not a valid image")
            os.replace(str(part), str(target))
            return "downloaded", relative_path, ""
        except Exception as exc:
            last_error = "{}: {}".format(type(exc).__name__, exc)
            if part.exists():
                part.unlink()
            if attempt < attempts:
                time.sleep(min(2 * attempt, 5))
    return "failed", relative_path, last_error


def main():
    args = parse_args()
    proxies = make_proxies(args.proxy)
    output_root = Path(args.output_root)
    rows = []
    with open(args.manifest, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                continue
            url, relative_path = line.split("\t", 1)
            rows.append((url, relative_path.replace("\\", "/")))

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {
            executor.submit(
                download_one,
                url,
                relative_path,
                output_root,
                args.timeout,
                args.attempts,
                proxies,
            ): relative_path
            for url, relative_path in rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            status, relative_path, error = future.result()
            counts[status] += 1
            if status == "failed":
                failures.append(
                    {"path": relative_path, "error": error}
                )
            if index % 50 == 0 or index == len(futures):
                print(
                    "processed {}/{} downloaded={} skipped={} failed={}".format(
                        index,
                        len(futures),
                        counts["downloaded"],
                        counts["skipped"],
                        counts["failed"],
                    ),
                    flush=True,
                )

    report = {
        "manifest": os.path.abspath(args.manifest),
        "output_root": os.path.abspath(args.output_root),
        "total": len(rows),
        **counts,
        "failures": failures,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(report_path), "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
