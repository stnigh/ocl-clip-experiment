import json
import math
import pickle
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DEFAULT_PROMPT_TEMPLATES = (
    "a photo of a {}",
    "an image of a {}",
    "a picture of a {}",
)


def clip_transform(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def _label_set(value):
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return set(int(item) for item in value)


def _normalize_box(box, width, height):
    if not box or len(box) < 4:
        return [0.0, 0.0, float(width), float(height)]
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    x1 = min(max(x1, 0.0), max(float(width) - 1.0, 0.0))
    y1 = min(max(y1, 0.0), max(float(height) - 1.0, 0.0))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return [0.0, 0.0, float(width), float(height)]
    return [x1, y1, x2, y2]


def _crop_image(image, box):
    left = max(0, int(math.floor(box[0])))
    top = max(0, int(math.floor(box[1])))
    right = min(image.width, int(math.ceil(box[2])))
    bottom = min(image.height, int(math.ceil(box[3])))
    if right <= left or bottom <= top:
        return image.copy()
    return image.crop((left, top, right, bottom))


def _affordance_names(affordance_data):
    names = []
    for item in affordance_data:
        if isinstance(item, str):
            names.append(item)
            continue
        words = item.get("word", [])
        if isinstance(words, str):
            names.append(words)
        elif words:
            names.append(words[0])
        else:
            names.append(item.get("define", "unknown").strip())
    return names


class OCLDataset(Dataset):
    def __init__(
        self,
        data_root,
        usable_manifest,
        annot_dir="data/resources",
        transform=None,
        image_size=224,
        max_samples=None,
    ):
        self.data_root = Path(data_root)
        self.annot_dir = Path(annot_dir)
        self.transform = transform or clip_transform(image_size)
        self.usable_manifest = Path(usable_manifest)

        self.object_names = self._load_json("OCL_class_object.json")
        self.attribute_names = self._load_json("OCL_class_attribute.json")
        self.affordance_data = self._load_json(
            "OCL_class_affordance.json"
        )
        self.affordance_names = _affordance_names(self.affordance_data)

        annotations = []
        class_counts = Counter()
        split_annotations = {}
        for split in ("val", "test"):
            with open(
                self.annot_dir / "OCL_annot_{}.pkl".format(split), "rb"
            ) as fp:
                split_images = pickle.load(fp)
            split_annotations[split] = split_images
            for image in split_images:
                for obj in image["objects"]:
                    class_counts[obj["obj"]] += 1

        self.top10 = class_counts.most_common(10)
        self.top10_names = [name for name, _ in self.top10]
        self.class_to_idx = {
            name: index for index, name in enumerate(self.top10_names)
        }
        self.top10_set = set(self.top10_names)

        usable_names = {
            line.strip()
            for line in self.usable_manifest.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
        self.attribute_size = len(self.attribute_names)
        self.affordance_size = len(self.affordance_names)

        for split, images in split_annotations.items():
            for image in images:
                if image["name"] not in usable_names:
                    continue
                top_objects = [
                    obj
                    for obj in image["objects"]
                    if obj["obj"] in self.top10_set
                ]
                if not top_objects:
                    continue
                annotations.append(
                    {
                        "name": image["name"],
                        "split": split,
                        "source": image.get(
                            "source", image["name"].split("/", 1)[0]
                        ),
                        "objects": top_objects,
                    }
                )

        annotations.sort(key=lambda item: item["name"])
        if max_samples is not None:
            annotations = annotations[: int(max_samples)]
        self.samples = annotations

        manifest_names = {item["name"] for item in self.samples}
        expected_names = (
            usable_names
            if max_samples is None
            else set(sorted(usable_names)[: int(max_samples)])
        )
        if manifest_names != expected_names:
            missing = sorted(expected_names - manifest_names)
            raise ValueError(
                "Manifest and annotation mismatch; missing examples: {}".format(
                    missing[:5]
                )
            )

        missing_files = [
            item["name"]
            for item in self.samples
            if not (self.data_root / Path(item["name"])).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                "Missing image files: {}".format(missing_files[:5])
            )

    def _load_json(self, name):
        with open(self.annot_dir / name, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = self.data_root / Path(sample["name"])
        with Image.open(str(path)) as image:
            image = image.convert("RGB")
            crops = []
            boxes = []
            object_indices = []
            labels = []
            attributes = []
            affordances = []

            for object_index, obj in enumerate(sample["objects"]):
                box = _normalize_box(
                    obj.get("box"), image.width, image.height
                )
                crop = _crop_image(image, box)
                crops.append(self.transform(crop))
                boxes.append(box)
                object_indices.append(object_index)
                labels.append(self.class_to_idx[obj["obj"]])

                attr = torch.zeros(self.attribute_size, dtype=torch.float32)
                aff = torch.zeros(
                    self.affordance_size, dtype=torch.float32
                )
                attr_indices = _label_set(obj.get("attr"))
                aff_indices = _label_set(obj.get("aff"))
                if attr_indices:
                    attr[torch.tensor(sorted(attr_indices), dtype=torch.long)] = 1.0
                if aff_indices:
                    aff[torch.tensor(sorted(aff_indices), dtype=torch.long)] = 1.0
                attributes.append(attr)
                affordances.append(aff)

            image_size = [image.width, image.height]

        if not crops:
            raise ValueError("No Top-10 objects found in {}".format(sample["name"]))

        return {
            "image_id": sample["name"],
            "split": sample["split"],
            "source": sample["source"],
            "image_size": image_size,
            "crops": torch.stack(crops, dim=0),
            "obj_labels": torch.tensor(labels, dtype=torch.long),
            "attributes": torch.stack(attributes, dim=0),
            "affordances": torch.stack(affordances, dim=0),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "object_indices": torch.tensor(
                object_indices, dtype=torch.long
            ),
        }

    def class_prompts(self, templates=DEFAULT_PROMPT_TEMPLATES):
        return {
            name: [template.format(name) for template in templates]
            for name in self.top10_names
        }

    def save_class_map(self, path):
        payload = {
            "top10": [
                {"id": index, "name": name, "instances": count}
                for index, (name, count) in enumerate(self.top10)
            ],
            "object_names": self.object_names,
            "attribute_names": self.attribute_names,
            "affordance_names": self.affordance_names,
            "prompt_templates": list(DEFAULT_PROMPT_TEMPLATES),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def ocl_collate_fn(batch):
    offsets = [0]
    crops = []
    labels = []
    attributes = []
    affordances = []
    for item in batch:
        count = item["crops"].shape[0]
        offsets.append(offsets[-1] + count)
        crops.append(item["crops"])
        labels.append(item["obj_labels"])
        attributes.append(item["attributes"])
        affordances.append(item["affordances"])

    return {
        "crops": torch.cat(crops, dim=0),
        "offsets": offsets,
        "image_ids": [item["image_id"] for item in batch],
        "splits": [item["split"] for item in batch],
        "sources": [item["source"] for item in batch],
        "image_sizes": [item["image_size"] for item in batch],
        "obj_labels": torch.cat(labels, dim=0),
        "attributes": torch.cat(attributes, dim=0),
        "affordances": torch.cat(affordances, dim=0),
        "boxes": [item["boxes"] for item in batch],
        "object_indices": [
            item["object_indices"] for item in batch
        ],
    }
