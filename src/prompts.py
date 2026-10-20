import json
from pathlib import Path


CLASS_TEMPLATES = (
    "a photo of a {}.",
    "an image of a {}.",
    "a picture of a {}.",
)
ATTRIBUTE_POSITIVE_TEMPLATES = (
    "an object that is {}.",
    "an object with the property {}.",
)
ATTRIBUTE_NEGATIVE_TEMPLATES = (
    "an object that is not {}.",
    "an object without the property {}.",
)
AFFORDANCE_POSITIVE_TEMPLATES = (
    "an object that can {}.",
    "an object used to {}.",
)
AFFORDANCE_NEGATIVE_TEMPLATES = (
    "an object that cannot {}.",
    "an object not used to {}.",
)


def load_class_map(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def ordered_top10(class_map):
    entries = sorted(class_map["top10"], key=lambda item: item["id"])
    expected_ids = list(range(len(entries)))
    if [item["id"] for item in entries] != expected_ids:
        raise ValueError("Top-10 ids must be contiguous and start at zero")
    return entries


def build_class_prompts(class_map, templates=CLASS_TEMPLATES):
    return [
        {
            "id": item["id"],
            "label": item["name"],
            "prompts": [
                template.format(item["name"]) for template in templates
            ],
        }
        for item in ordered_top10(class_map)
    ]


def build_binary_prompts(
    labels,
    positive_templates,
    negative_templates,
):
    result = []
    for index, label in enumerate(labels):
        result.append(
            {
                "id": index,
                "label": label,
                "positive": [
                    template.format(label)
                    for template in positive_templates
                ],
                "negative": [
                    template.format(label)
                    for template in negative_templates
                ],
            }
        )
    return result


def build_prompt_bank(
    class_map_path,
    class_templates=CLASS_TEMPLATES,
    attribute_positive_templates=ATTRIBUTE_POSITIVE_TEMPLATES,
    attribute_negative_templates=ATTRIBUTE_NEGATIVE_TEMPLATES,
    affordance_positive_templates=AFFORDANCE_POSITIVE_TEMPLATES,
    affordance_negative_templates=AFFORDANCE_NEGATIVE_TEMPLATES,
):
    class_map = load_class_map(class_map_path)
    attribute_prompts = build_binary_prompts(
        class_map["attribute_names"],
        attribute_positive_templates,
        attribute_negative_templates,
    )
    affordance_prompts = build_binary_prompts(
        class_map["affordance_names"],
        affordance_positive_templates,
        affordance_negative_templates,
    )
    return {
        "classes": build_class_prompts(class_map, class_templates),
        "attributes": attribute_prompts,
        "affordances": affordance_prompts,
    }


def save_prompt_bank(prompt_bank, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(prompt_bank, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def flatten_group_prompts(groups, key="prompts"):
    texts = []
    group_sizes = []
    labels = []
    for group in groups:
        prompts = group[key]
        texts.extend(prompts)
        group_sizes.append(len(prompts))
        labels.append(group["label"])
    return texts, group_sizes, labels


def flatten_binary_prompts(groups):
    positive = []
    negative = []
    positive_sizes = []
    negative_sizes = []
    labels = []
    for group in groups:
        positive.extend(group["positive"])
        negative.extend(group["negative"])
        positive_sizes.append(len(group["positive"]))
        negative_sizes.append(len(group["negative"]))
        labels.append(group["label"])
    return {
        "positive": positive,
        "negative": negative,
        "positive_sizes": positive_sizes,
        "negative_sizes": negative_sizes,
        "labels": labels,
    }

