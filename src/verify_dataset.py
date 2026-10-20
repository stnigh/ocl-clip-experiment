import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit the OCL Top-10 experimental dataset."
    )
    parser.add_argument("--annot-dir", default="data/resources")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--usable-manifest", required=True)
    parser.add_argument("--excluded-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--min-size", type=int, default=32)
    parser.add_argument("--skip-image-check", action="store_true")
    return parser.parse_args()


def read_lines(path):
    with open(path, "r", encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def read_excluded(path):
    excluded = {}
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            excluded[fields[0]] = fields[1] if len(fields) > 1 else "unknown"
    return excluded


def load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def label_indices(value):
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return set(int(item) for item in value)


def image_status(path, min_size):
    if not path.is_file():
        return "missing", None
    try:
        with Image.open(str(path)) as image:
            image.verify()
        with Image.open(str(path)) as image:
            image.load()
            if image.width < min_size or image.height < min_size:
                return "too_small", list(image.size)
            return "valid", list(image.size)
    except Exception as exc:
        return "corrupt", str(exc)


def main():
    args = parse_args()
    annot_dir = Path(args.annot_dir)
    data_root = Path(args.data_root)
    usable = set(read_lines(args.usable_manifest))
    excluded = read_excluded(args.excluded_manifest)
    required = usable | set(excluded)

    classes = {
        "object": load_json(annot_dir / "OCL_class_object.json"),
        "attribute": load_json(annot_dir / "OCL_class_attribute.json"),
        "affordance": load_json(annot_dir / "OCL_class_affordance.json"),
    }

    split_annotations = {}
    image_records = {}
    object_counts = Counter()
    for split in ("val", "test"):
        with open(annot_dir / "OCL_annot_{}.pkl".format(split), "rb") as fp:
            split_annotations[split] = pickle.load(fp)
        for image in split_annotations[split]:
            for obj in image["objects"]:
                object_counts[obj["obj"]] += 1

    top10 = [name for name, _ in object_counts.most_common(10)]
    top10_set = set(top10)
    for split, images in split_annotations.items():
        for image in images:
            top_objects = [
                obj for obj in image["objects"] if obj["obj"] in top10_set
            ]
            if top_objects:
                image_records[image["name"]] = {
                    "split": split,
                    "image": image,
                    "top_objects": top_objects,
                }

    missing_from_manifest = sorted(set(image_records) - required)
    extra_in_manifest = sorted(required - set(image_records))
    if missing_from_manifest or extra_in_manifest:
        raise ValueError(
            "Manifest mismatch: missing_from_manifest={}, extra_in_manifest={}".format(
                len(missing_from_manifest), len(extra_in_manifest)
            )
        )

    by_split = {
        split: Counter(
            images_total=0,
            images_usable=0,
            images_excluded=0,
            top10_objects_total=0,
            top10_objects_usable=0,
            top10_objects_excluded=0,
        )
        for split in ("val", "test")
    }
    by_source = defaultdict(Counter)
    by_class = {
        name: Counter(
            images_total=0,
            images_usable=0,
            images_excluded=0,
            objects_total=0,
            objects_usable=0,
            objects_excluded=0,
        )
        for name in top10
    }
    attribute_counts = Counter()
    affordance_counts = Counter()
    status_counts = Counter()
    status_mismatches = []

    for name, record in image_records.items():
        split = record["split"]
        image = record["image"]
        objects = record["top_objects"]
        source = name.split("/", 1)[0]
        is_usable = name in usable
        classes_in_image = {obj["obj"] for obj in objects}

        status = "not_checked"
        size = None
        if not args.skip_image_check:
            status, size = image_status(data_root / Path(name), args.min_size)
            status_counts[status] += 1
            if is_usable and status != "valid":
                status_mismatches.append([name, status, size])
            if not is_usable and status == "valid":
                status_mismatches.append([name, "usable_but_excluded", size])

        target = by_split[split]
        target["images_total"] += 1
        target["top10_objects_total"] += len(objects)
        by_source[source]["images_total"] += 1
        by_source[source]["objects_total"] += len(objects)
        if is_usable:
            target["images_usable"] += 1
            target["top10_objects_usable"] += len(objects)
            by_source[source]["images_usable"] += 1
            by_source[source]["objects_usable"] += len(objects)
        else:
            target["images_excluded"] += 1
            target["top10_objects_excluded"] += len(objects)
            by_source[source]["images_excluded"] += 1
            by_source[source]["objects_excluded"] += len(objects)

        for cls in classes_in_image:
            by_class[cls]["images_total"] += 1
            if is_usable:
                by_class[cls]["images_usable"] += 1
            else:
                by_class[cls]["images_excluded"] += 1
        for obj in objects:
            cls = obj["obj"]
            by_class[cls]["objects_total"] += 1
            if is_usable:
                by_class[cls]["objects_usable"] += 1
                attribute_counts.update(label_indices(obj.get("attr")))
                affordance_counts.update(label_indices(obj.get("aff")))
            else:
                by_class[cls]["objects_excluded"] += 1

    attribute_names = [
        classes["attribute"][index]
        for index in sorted(attribute_counts)
    ]
    affordance_names = [
        classes["affordance"][index]
        for index in sorted(affordance_counts)
    ]
    result = {
        "data_root": str(data_root),
        "usable_manifest": str(args.usable_manifest),
        "excluded_manifest": str(args.excluded_manifest),
        "top10_classes": [
            {"class": name, "instances": object_counts[name]} for name in top10
        ],
        "images": {
            "required": len(required),
            "usable": len(usable),
            "excluded": len(excluded),
            "valid_coverage_pct": round(100 * len(usable) / len(required), 2),
        },
        "top10_objects": {
            "total": sum(item["top10_objects_total"] for item in by_split.values()),
            "usable": sum(item["top10_objects_usable"] for item in by_split.values()),
            "excluded": sum(item["top10_objects_excluded"] for item in by_split.values()),
        },
        "by_split": {key: dict(value) for key, value in by_split.items()},
        "by_source": {key: dict(value) for key, value in by_source.items()},
        "by_class": {key: dict(value) for key, value in by_class.items()},
        "excluded_reasons": dict(Counter(excluded.values())),
        "image_status_counts": dict(status_counts),
        "status_mismatches": status_mismatches[:100],
        "attribute_counts": dict(attribute_counts),
        "attribute_names": attribute_names,
        "affordance_counts": dict(affordance_counts),
        "affordance_names": affordance_names,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# OCL 数据对照审计",
        "",
        "## 总览",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        "| Top-10 图片总数 | {} |".format(len(required)),
        "| 可用图片 | {} |".format(len(usable)),
        "| 排除图片 | {} |".format(len(excluded)),
        "| Top-10 对象总数 | {} |".format(result["top10_objects"]["total"]),
        "| 可用对象 | {} |".format(result["top10_objects"]["usable"]),
        "| 排除对象 | {} |".format(result["top10_objects"]["excluded"]),
        "",
        "## split 对照",
        "",
        "| split | 图片总数 | 可用图片 | 排除图片 | 对象总数 | 可用对象 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("val", "test"):
        value = by_split[split]
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                split,
                value["images_total"],
                value["images_usable"],
                value["images_excluded"],
                value["top10_objects_total"],
                value["top10_objects_usable"],
            )
        )
    lines.extend(
        [
            "",
            "## 类别对照",
            "",
            "| 类别 | 对象总数 | 可用对象 | 可用图片 | 排除图片 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in top10:
        value = by_class[name]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                name,
                value["objects_total"],
                value["objects_usable"],
                value["images_usable"],
                value["images_excluded"],
            )
        )
    lines.extend(
        [
            "",
            "## 排除原因",
            "",
            "| 原因 | 数量 |",
            "|---|---:|",
        ]
    )
    for reason, count in Counter(excluded.values()).items():
        lines.append("| {} | {} |".format(reason, count))
    lines.extend(
        [
            "",
            "## 图像检查",
            "",
            "| 状态 | 数量 |",
            "|---|---:|",
        ]
    )
    for status, count in status_counts.items():
        lines.append("| {} | {} |".format(status, count))

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["images"], ensure_ascii=False, indent=2))
    print(json.dumps(result["top10_objects"], ensure_ascii=False, indent=2))
    if status_mismatches:
        print(
            "STATUS_MISMATCHES={}".format(len(status_mismatches)),
            flush=True,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
