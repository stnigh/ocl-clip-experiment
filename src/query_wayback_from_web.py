import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


WAYBACK_URL = "https://archive.org/wayback/available"
CDX_URL = "https://web.archive.org/cdx/search/cdx"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find Wayback image snapshots for missing OCL web URLs."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--proxy", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--api", choices=("availability", "cdx"), default="availability"
    )
    return parser.parse_args()


def make_proxies(proxy):
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def query_availability(url, relative_path, proxies, timeout, attempts):
    params = {"url": url}
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                WAYBACK_URL,
                params=params,
                headers=HEADERS,
                proxies=proxies,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            closest = data.get("archived_snapshots", {}).get("closest")
            if not closest or not closest.get("available"):
                return {
                    "path": relative_path,
                    "original_url": url,
                    "status": "not_found",
                    "snapshot_url": "",
                    "timestamp": "",
                    "mimetype": "",
                    "length": 0,
                }

            timestamp = closest.get("timestamp", "")
            return {
                "path": relative_path,
                "original_url": url,
                "status": "found",
                "snapshot_url": (
                    "https://web.archive.org/web/{}id_/{}".format(
                        timestamp, url
                    )
                ),
                "timestamp": timestamp,
                "mimetype": "",
                "length": 0,
            }
        except Exception as exc:
            last_error = "{}: {}".format(type(exc).__name__, exc)
            if attempt < attempts:
                time.sleep(min(2 * attempt, 6))

    return {
        "path": relative_path,
        "original_url": url,
        "status": "error",
        "snapshot_url": "",
        "timestamp": "",
        "mimetype": "",
        "length": 0,
        "error": last_error,
    }


def query_cdx(url, relative_path, proxies, timeout, attempts):
    params = {
        "url": url,
        "output": "json",
        "filter": ["statuscode:200", "mimetype:image/.*"],
        "fl": "timestamp,original,statuscode,mimetype,length",
        "limit": "-5",
        "matchType": "exact",
    }
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                CDX_URL,
                params=params,
                headers=HEADERS,
                proxies=proxies,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            if not data:
                return {
                    "path": relative_path,
                    "original_url": url,
                    "status": "not_found",
                    "snapshot_url": "",
                    "timestamp": "",
                    "mimetype": "",
                    "length": 0,
                }

            rows = data[1:] if data[0][0] == "timestamp" else data
            image_rows = [
                (row[0], row[1], row[3], int(row[4] or 0))
                for row in rows
                if len(row) >= 5
                and row[2] == "200"
                and row[3].startswith("image/")
            ]
            if not image_rows:
                return {
                    "path": relative_path,
                    "original_url": url,
                    "status": "not_found",
                    "snapshot_url": "",
                    "timestamp": "",
                    "mimetype": "",
                    "length": 0,
                }

            timestamp, original, mimetype, length = max(
                image_rows, key=lambda item: (item[0], item[3])
            )
            return {
                "path": relative_path,
                "original_url": url,
                "status": "found",
                "snapshot_url": (
                    "https://web.archive.org/web/{}id_/{}".format(
                        timestamp, original
                    )
                ),
                "timestamp": timestamp,
                "mimetype": mimetype,
                "length": length,
            }
        except Exception as exc:
            last_error = "{}: {}".format(type(exc).__name__, exc)
            if attempt < attempts:
                time.sleep(min(2 * attempt, 6))

    return {
        "path": relative_path,
        "original_url": url,
        "status": "error",
        "snapshot_url": "",
        "timestamp": "",
        "mimetype": "",
        "length": 0,
        "error": last_error,
    }


def query_one(api, url, relative_path, proxies, timeout, attempts):
    if api == "cdx":
        return query_cdx(
            url, relative_path, proxies, timeout, attempts
        )
    return query_availability(
        url, relative_path, proxies, timeout, attempts
    )


def main():
    args = parse_args()
    proxies = make_proxies(args.proxy)
    rows = []
    with open(args.manifest, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                continue
            url, relative_path = line.split("\t", 1)
            rows.append((url, relative_path.replace("\\", "/")))

    results = []
    report_path = Path(args.report)
    snapshot_manifest = report_path.with_name(
        report_path.stem + "_snapshots.tsv"
    )

    def write_outputs():
        ordered = sorted(results, key=lambda item: item["path"])
        found_items = [item for item in ordered if item["status"] == "found"]
        report = {
            "total": len(rows),
            "processed": len(ordered),
            "found": len(found_items),
            "not_found": sum(
                item["status"] == "not_found" for item in ordered
            ),
            "errors": sum(item["status"] == "error" for item in ordered),
            "results": ordered,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with open(snapshot_manifest, "w", encoding="utf-8") as fp:
            for item in found_items:
                fp.write(
                    "{}\t{}\n".format(item["snapshot_url"], item["path"])
                )
        return report

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {
            executor.submit(
                query_one,
                args.api,
                url,
                relative_path,
                proxies,
                args.timeout,
                args.attempts,
            ): relative_path
            for url, relative_path in rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if args.delay > 0:
                time.sleep(args.delay)
            if index % 25 == 0 or index == len(futures):
                write_outputs()
                found = sum(item["status"] == "found" for item in results)
                errors = sum(item["status"] == "error" for item in results)
                print(
                    "processed {}/{} found={} errors={}".format(
                        index, len(futures), found, errors
                    ),
                    flush=True,
                )

    report = write_outputs()

    print(
        json.dumps(
            {
                "total": report["total"],
                "found": report["found"],
                "not_found": report["not_found"],
                "errors": report["errors"],
                "report": str(report_path),
                "snapshot_manifest": str(snapshot_manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
