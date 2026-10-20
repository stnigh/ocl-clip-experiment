import argparse
import json
import os
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract only the COCO images required by an OCL manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-zip", required=True)
    parser.add_argument("--val-zip", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def load_required_members(manifest_path):
    required = {"train2014": set(), "val2014": set()}
    with open(manifest_path, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip().replace("\\", "/")
            if not line.startswith("COCO/"):
                continue
            member = line[len("COCO/") :]
            split = member.split("/", 1)[0]
            if split not in required:
                raise ValueError("Unexpected COCO split: {}".format(member))
            if ".." in Path(member).parts:
                raise ValueError("Unsafe manifest path: {}".format(member))
            required[split].add(member)
    return required


def extract_split(zip_path, output_root, required_members):
    zip_path = Path(zip_path)
    output_root = Path(output_root)
    if not zip_path.is_file():
        raise FileNotFoundError(str(zip_path))

    extracted = 0
    skipped = 0
    missing = []
    with zipfile.ZipFile(str(zip_path), "r") as archive:
        available = {
            info.filename.replace("\\", "/")
            for info in archive.infolist()
            if not info.is_dir()
        }
        missing = sorted(required_members - available)
        wanted = required_members & available

        for index, info in enumerate(archive.infolist(), start=1):
            member = info.filename.replace("\\", "/")
            if info.is_dir() or member not in wanted:
                continue

            target = output_root / Path(member)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size > 0:
                skipped += 1
                continue

            archive.extract(info, str(output_root))
            extracted += 1
            if extracted % 500 == 0:
                print("{}: extracted {}".format(zip_path.name, extracted), flush=True)

    return {
        "archive": str(zip_path),
        "required": len(required_members),
        "available": len(wanted),
        "extracted": extracted,
        "skipped_existing": skipped,
        "missing_in_archive": missing,
    }


def main():
    args = parse_args()
    required = load_required_members(args.manifest)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    reports = {
        "train2014": extract_split(
            args.train_zip, output_root, required["train2014"]
        ),
        "val2014": extract_split(args.val_zip, output_root, required["val2014"]),
    }

    report = {
        "manifest": os.path.abspath(args.manifest),
        "output_root": os.path.abspath(args.output_root),
        "splits": reports,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(report_path), "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    total_missing = sum(
        len(split["missing_in_archive"]) for split in reports.values()
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if total_missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
