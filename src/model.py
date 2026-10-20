import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .prompts import flatten_binary_prompts, flatten_group_prompts


DEFAULT_CLIP_CACHE = os.environ.get(
    "OCL_CLIP_CACHE", r"E:\OCL_Data\models\clip"
)


def choose_device(preferred="cuda"):
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def freeze_module(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


class OpenAICLIPFeatureExtractor:
    def __init__(self, model, preprocess, device, model_name):
        self.model = model.to(device)
        self.preprocess = preprocess
        self.device = device
        self.model_name = model_name
        self.feature_dim = int(self.model.text_projection.shape[1])

    @torch.no_grad()
    def encode_images(self, images):
        images = images.to(self.device)
        features = self.model.encode_image(images)
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=64):
        import clip

        outputs = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            tokens = clip.tokenize(chunk, truncate=True).to(self.device)
            features = self.model.encode_text(tokens)
            outputs.append(F.normalize(features.float(), dim=-1))
        if not outputs:
            return torch.empty(
                0, self.feature_dim, device=self.device
            )
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def encode_text_groups(
        self, groups, key="prompts", text_batch_size=64
    ):
        texts, sizes, labels = flatten_group_prompts(groups, key=key)
        embeddings = self.encode_texts(
            texts, batch_size=text_batch_size
        )
        grouped = []
        offset = 0
        for size in sizes:
            group_embedding = embeddings[offset : offset + size].mean(
                dim=0
            )
            grouped.append(F.normalize(group_embedding, dim=0))
            offset += size
        if not grouped:
            return (
                torch.empty(
                    0, self.feature_dim, device=self.device
                ),
                labels,
            )
        return torch.stack(grouped, dim=0), labels

    @torch.no_grad()
    def encode_binary_groups(
        self, groups, text_batch_size=64
    ):
        flat = flatten_binary_prompts(groups)
        positive = self.encode_texts(
            flat["positive"], batch_size=text_batch_size
        )
        negative = self.encode_texts(
            flat["negative"], batch_size=text_batch_size
        )

        def average_groups(embeddings, sizes):
            outputs = []
            offset = 0
            for size in sizes:
                output = embeddings[offset : offset + size].mean(dim=0)
                outputs.append(F.normalize(output, dim=0))
                offset += size
            return torch.stack(outputs, dim=0)

        return {
            "positive": average_groups(
                positive, flat["positive_sizes"]
            ),
            "negative": average_groups(
                negative, flat["negative_sizes"]
            ),
            "labels": flat["labels"],
        }


def load_openai_clip(
    model_name="ViT-B/32",
    device="cuda",
    download_root=None,
    require_cuda=False,
):
    try:
        import clip
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI CLIP is not installed. Install the official clip "
            "package before loading the model."
        ) from exc

    resolved_device = choose_device(device)
    if require_cuda and resolved_device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available")

    download_root = Path(download_root or DEFAULT_CLIP_CACHE)
    download_root.mkdir(parents=True, exist_ok=True)
    model, preprocess = clip.load(
        model_name,
        device=resolved_device,
        download_root=str(download_root),
    )
    freeze_module(model)
    return OpenAICLIPFeatureExtractor(
        model=model,
        preprocess=preprocess,
        device=resolved_device,
        model_name=model_name,
    )


def get_clip_summary(feature_extractor):
    model = feature_extractor.model
    return {
        "model_name": feature_extractor.model_name,
        "device": str(feature_extractor.device),
        "feature_dim": feature_extractor.feature_dim,
        "input_resolution": int(model.visual.input_resolution),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
