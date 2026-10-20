import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CDX_URL = "https://web.archive.org/cdx/search/cdx"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query Wayback CDX with a hard per-request timeout."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--proxy", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-time", type=int, default=20)
    return parser.parse_args()


def query_one(url, relative_path, proxy, max_time):
    command = [
        "curl.exe",
        "-sS",
        "-G",
        "--connect-timeout",
        "10",
        "--max-time",
        str(max_time),
    ]
    if proxy:
        command.extend(["--proxy", proxy])
    command.extend(
        [
            CDX_URL,
            "--data-urlencode",
            "url={}".format(url),
            "--data-urlencode",
            "output=json",
            "--data-urlencode",
            "filter=statuscode:200",
            "--data-urlencode",
            "filter=mimetype:image/.*",
            "--data-urlencode",
            "fl=timestamp,original,statuscode,mimetype,length",
            "--data-urlencode",
            "limit=-5",
            "--data-urlencode",
            "matchType=exact",
        ]
    )
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max_time + 5,
            check=False,
        )
        if process.returncode != 0:
            return {
                "path": relative_path,
                "original_url": url,
                "status": "error",
                "snapshot_url": "",
                "timestamp": "",
                "mimetype": "",
                "length": 0,
                "error": process.stderr.strip()
                or "curl exit {}".format(process.returncode),
            }
        raw = process.stdout.strip()
        if not raw:
            data = []
        else:
            data = json.loads(raw)
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
        images = [
            (row[0], row[1], row[3], int(row[4] or 0))
            for row in rows
            if len(row) >= 5
            and row[2] == "200"
            and row[3].startswith("image/")
        ]
        if not images:
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
            images, key=lambda item: (item[0], item[3])
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
        return {
            "path": relative_path,
            "original_url": url,
            "status": "error",
            "snapshot_url": "",
            "timestamp": "",
            "mimetype": "",
            "length": 0,
            "error": "{}: {}".format(type(exc).__name__, exc),
        }


def main():
    args = parse_args()
    rows = []
    with open(args.manifest, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                continue
            url, relative_path = line.split("\t", 1)
            rows.append((url, relative_path.replace("\\", "/")))

    report_path = Path(args.report)
    snapshot_manifest = report_path.with_name(
        report_path.stem + "_snapshots.tsv"
    )
    results = []

    def write_outputs():
        ordered = sorted(results, key=lambda item: item["path"])
        found = [item for item in ordered if item["status"] == "found"]
        report = {
            "total": len(rows),
            "processed": len(ordered),
            "found": len(found),
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
            for item in found:
                fp.write(
                    "{}\t{}\n".format(item["snapshot_url"], item["path"])
                )
        return report

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {
            executor.submit(
                query_one, url, relative_path, args.proxy, args.max_time
            ): relative_path
            for url, relative_path in rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 25 == 0 or index == len(futures):
                report = write_outputs()
                print(
                    "processed {}/{} found={} errors={}".format(
                        index,
                        len(futures),
                        report["found"],
                        report["errors"],
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


if __name__ == "__main__":
    main()
