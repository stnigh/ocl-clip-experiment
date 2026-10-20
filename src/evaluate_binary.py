import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .dataset import OCLDataset, ocl_collate_fn
from .infer import select_balanced_indices
from .model import get_clip_summary, load_openai_clip
from .prompts import build_prompt_bank


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate attribute and affordance AP/mAP with CLIP."
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
        "--scores-output", default="outputs/binary_scores.npz"
    )
    parser.add_argument(
        "--metrics-output", default="outputs/binary_metrics.json"
    )
    parser.add_argument(
        "--labels-output", default="outputs/binary_label_ap.json"
    )
    return parser.parse_args()


def expand_object_values(values, offsets):
    expanded = []
    for index, value in enumerate(values):
        expanded.extend([value] * (offsets[index + 1] - offsets[index]))
    return expanded


def compute_ap_table(scores, targets, labels):
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    ap = np.full(targets.shape[1], np.nan, dtype=np.float64)
    for index in range(targets.shape[1]):
        truth = targets[:, index] > 0
        if truth.sum() == 0:
            continue
        ap[index] = (
            average_precision_score(truth.astype(np.int32), scores[:, index])
            * 100.0
        )
    return ap


def summarize_ap(ap, labels):
    values = []
    for index, (label, value) in enumerate(zip(labels, ap)):
        if not np.isnan(value):
            values.append(
                {"id": index, "label": label, "ap": float(value)}
            )
    values.sort(key=lambda item: item["ap"])
    return {
        "mean_ap": float(np.nanmean(ap)),
        "valid_labels": len(values),
        "total_labels": len(labels),
        "lowest": values[:10],
        "highest": list(reversed(values[-10:])),
    }


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
    attribute_text = extractor.encode_binary_groups(
        bank["attributes"], text_batch_size=64
    )
    affordance_text = extractor.encode_binary_groups(
        bank["affordances"], text_batch_size=64
    )

    attribute_scores = []
    attribute_targets = []
    affordance_scores = []
    affordance_targets = []
    object_sources = []
    object_splits = []
    object_class_ids = []
    image_count = 0

    for batch in tqdm(
        loader,
        total=len(loader),
        desc="binary inference",
        ncols=80,
        mininterval=5,
    ):
        image_features = extractor.encode_images(batch["crops"])
        attr_scores = (
            image_features @ attribute_text["positive"].t()
            - image_features @ attribute_text["negative"].t()
        )
        aff_scores = (
            image_features @ affordance_text["positive"].t()
            - image_features @ affordance_text["negative"].t()
        )
        attribute_scores.append(
            attr_scores.cpu().numpy().astype(np.float32)
        )
        affordance_scores.append(
            aff_scores.cpu().numpy().astype(np.float32)
        )
        attribute_targets.append(
            batch["attributes"].cpu().numpy().astype(np.int8)
        )
        affordance_targets.append(
            batch["affordances"].cpu().numpy().astype(np.int8)
        )
        object_sources.extend(
            expand_object_values(batch["sources"], batch["offsets"])
        )
        object_splits.extend(
            expand_object_values(batch["splits"], batch["offsets"])
        )
        object_class_ids.extend(batch["obj_labels"].cpu().tolist())
        image_count += len(batch["image_ids"])

    attr_scores = np.concatenate(attribute_scores, axis=0)
    attr_targets = np.concatenate(attribute_targets, axis=0)
    aff_scores = np.concatenate(affordance_scores, axis=0)
    aff_targets = np.concatenate(affordance_targets, axis=0)
    object_sources = np.asarray(object_sources)
    object_splits = np.asarray(object_splits)
    object_class_ids = np.asarray(object_class_ids, dtype=np.int64)

    attr_ap = compute_ap_table(
        attr_scores, attr_targets, attribute_text["labels"]
    )
    aff_ap = compute_ap_table(
        aff_scores, aff_targets, affordance_text["labels"]
    )

    source_metrics = {}
    split_metrics = {}
    for source in sorted(set(object_sources.tolist())):
        mask = object_sources == source
        source_metrics[source] = {
            "objects": int(mask.sum()),
            "attribute_map": float(
                np.nanmean(
                    compute_ap_table(
                        attr_scores[mask],
                        attr_targets[mask],
                        attribute_text["labels"],
                    )
                )
            ),
            "affordance_map": float(
                np.nanmean(
                    compute_ap_table(
                        aff_scores[mask],
                        aff_targets[mask],
                        affordance_text["labels"],
                    )
                )
            ),
        }
    for split in sorted(set(object_splits.tolist())):
        mask = object_splits == split
        split_metrics[split] = {
            "objects": int(mask.sum()),
            "attribute_map": float(
                np.nanmean(
                    compute_ap_table(
                        attr_scores[mask],
                        attr_targets[mask],
                        attribute_text["labels"],
                    )
                )
            ),
            "affordance_map": float(
                np.nanmean(
                    compute_ap_table(
                        aff_scores[mask],
                        aff_targets[mask],
                        affordance_text["labels"],
                    )
                )
            ),
        }

    attribute_summary = summarize_ap(
        attr_ap, attribute_text["labels"]
    )
    affordance_summary = summarize_ap(
        aff_ap, affordance_text["labels"]
    )
    metrics = {
        "model": get_clip_summary(extractor),
        "images": image_count,
        "objects": int(attr_scores.shape[0]),
        "score_method": "positive_minus_negative_cosine",
        "attributes": attribute_summary,
        "affordances": affordance_summary,
        "by_source": source_metrics,
        "by_split": split_metrics,
    }

    scores_path = Path(args.scores_output)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(scores_path),
        attribute_scores=attr_scores,
        attribute_targets=attr_targets,
        affordance_scores=aff_scores,
        affordance_targets=aff_targets,
        object_sources=object_sources,
        object_splits=object_splits,
        object_class_ids=object_class_ids,
    )

    labels_payload = {
        "attributes": [
            {
                "id": index,
                "label": label,
                "ap": None if np.isnan(value) else float(value),
            }
            for index, (label, value) in enumerate(
                zip(attribute_text["labels"], attr_ap)
            )
        ],
        "affordances": [
            {
                "id": index,
                "label": label,
                "ap": None if np.isnan(value) else float(value),
            }
            for index, (label, value) in enumerate(
                zip(affordance_text["labels"], aff_ap)
            )
        ],
    }
    metrics_path = Path(args.metrics_output)
    labels_path = Path(args.labels_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps(labels_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
