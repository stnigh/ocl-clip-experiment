import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .dataset import OCLDataset, ocl_collate_fn
from .model import get_clip_summary, load_openai_clip
from .prompts import build_prompt_bank


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OCL CLIP object-classification inference."
    )
    parser.add_argument("--data-root", default=r"E:\OCL_Data\datasets")
    parser.add_argument(
        "--usable-manifest",
        default=(
            r"C:\Users\17031\Desktop\OCL_Workspace\manifests"
            r"\OCL_valtest_top10_usable_files.txt"
        ),
    )
    parser.add_argument("--annot-dir", default="data/resources")
    parser.add_argument("--class-map", default="outputs/class_map.json")
    parser.add_argument("--model-name", default="ViT-B/16")
    parser.add_argument(
        "--download-root", default=r"C:\Users\17031\.cache\clip"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--predictions", default="outputs/smoke_predictions.jsonl"
    )
    parser.add_argument(
        "--metrics", default="outputs/smoke_metrics.json"
    )
    parser.add_argument("--config", default="outputs/smoke_config.json")
    return parser.parse_args()


def select_balanced_indices(dataset, limit):
    groups = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        groups[(sample["source"], sample["split"])].append(index)

    source_priority = {"COCO": 0, "ImageNet": 1, "from_web": 2}
    keys = sorted(
        groups,
        key=lambda key: (
            source_priority.get(key[0], 99),
            key[1],
        ),
    )
    cursors = {key: 0 for key in keys}
    selected = []
    while len(selected) < limit:
        added = False
        for key in keys:
            cursor = cursors[key]
            if cursor >= len(groups[key]):
                continue
            selected.append(groups[key][cursor])
            cursors[key] += 1
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def object_scores(scores, labels, class_names, top_k=5):
    k = min(top_k, scores.shape[1])
    values, indices = scores.topk(k, dim=1)
    values = values.cpu()
    indices = indices.cpu()
    labels = labels.cpu()

    records = []
    for row in range(scores.shape[0]):
        predicted = indices[row].tolist()
        predicted_scores = values[row].tolist()
        records.append(
            {
                "true_id": int(labels[row]),
                "true_name": class_names[int(labels[row])],
                "topk_ids": predicted,
                "topk_names": [class_names[index] for index in predicted],
                "topk_scores": predicted_scores,
            }
        )
    return records


def main():
    args = parse_args()
    extractor = load_openai_clip(
        model_name=args.model_name,
        device=args.device,
        download_root=args.download_root,
        require_cuda=(args.device == "cuda"),
    )
    base_dataset = OCLDataset(
        data_root=args.data_root,
        usable_manifest=args.usable_manifest,
        annot_dir=args.annot_dir,
        transform=extractor.preprocess,
    )
    indices = select_balanced_indices(base_dataset, args.limit)
    dataset = Subset(base_dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ocl_collate_fn,
    )

    bank = build_prompt_bank(args.class_map)
    class_features, class_names = extractor.encode_text_groups(
        bank["classes"], text_batch_size=32
    )
    if class_names != base_dataset.top10_names:
        raise ValueError("CLIP class prompt order does not match dataset labels")

    predictions_path = Path(args.predictions)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    hit_counts = Counter()
    object_count = 0
    image_count = 0
    source_counts = Counter()
    split_counts = Counter()

    with open(predictions_path, "w", encoding="utf-8") as fp:
        for batch in tqdm(
            loader,
            total=len(loader),
            desc="CLIP inference",
            ncols=80,
            mininterval=5,
        ):
            image_features = extractor.encode_images(batch["crops"])
            scores = image_features @ class_features.t()
            records = object_scores(
                scores, batch["obj_labels"], class_names, top_k=5
            )
            object_count += len(records)

            for metric_k in (1, 3, 5):
                hit_counts[metric_k] += sum(
                    record["true_id"] in record["topk_ids"][:metric_k]
                    for record in records
                )

            image_count += len(batch["image_ids"])
            source_counts.update(batch["sources"])
            split_counts.update(batch["splits"])

            for image_index, image_id in enumerate(batch["image_ids"]):
                begin = batch["offsets"][image_index]
                end = batch["offsets"][image_index + 1]
                objects = []
                for record_index in range(begin, end):
                    record = dict(records[record_index])
                    record["object_index"] = int(
                        batch["object_indices"][image_index][
                            record_index - begin
                        ]
                    )
                    record["box"] = batch["boxes"][image_index][
                        record_index - begin
                    ].tolist()
                    objects.append(record)
                item = {
                    "image_id": image_id,
                    "split": batch["splits"][image_index],
                    "source": batch["sources"][image_index],
                    "image_size": batch["image_sizes"][image_index],
                    "objects": objects,
                }
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics = {
        "model": get_clip_summary(extractor),
        "limit": args.limit,
        "sampled_images": image_count,
        "sampled_objects": object_count,
        "source_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "top1_accuracy": (
            hit_counts[1] / object_count if object_count else 0.0
        ),
        "top3_accuracy": (
            hit_counts[3] / object_count if object_count else 0.0
        ),
        "top5_accuracy": (
            hit_counts[5] / object_count if object_count else 0.0
        ),
        "correct_top1": hit_counts[1],
        "correct_top3": hit_counts[3],
        "correct_top5": hit_counts[5],
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config = {
        "model_name": args.model_name,
        "device": str(extractor.device),
        "batch_size": args.batch_size,
        "limit": args.limit,
        "num_workers": args.num_workers,
        "usable_manifest": args.usable_manifest,
        "data_root": args.data_root,
        "class_map": args.class_map,
        "class_names": class_names,
    }
    config_path = Path(args.config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
