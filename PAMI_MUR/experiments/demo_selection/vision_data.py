"""Frozen-CLIP feature extraction and caching for the R110 demonstration-selection family.

The second task family replaces traffic sensors by *labelled training examples*
("demonstrations").  The anchor is a query image batch, the candidate pool is a
set of demonstration images, and the frozen predictor is a small in-context
transformer (see ``incontext_predictor.py``).

This module only handles the observable side of the protocol: it encodes every
image of a benchmark once with frozen CLIP ``ViT-L-14`` (openai weights) and
caches the L2-normalised embeddings as ``float16`` ``.npy`` files.  Later runs
never touch image files again.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "openai"
CLIP_CACHE_DIR = "/root/.cache/clip"
EMBED_DIM = 768
EUROSAT_TRAIN_PER_CLASS = 1620  # stratified 60/20/20-style split (1620/540 per class)
EUROSAT_TEST_PER_CLASS = 540


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    classes: int
    source: str
    note: str


SPECS: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec("cifar10", 10, "torchvision.datasets.CIFAR10", "official 50k/10k split"),
    "cifar100": DatasetSpec("cifar100", 100, "torchvision.datasets.CIFAR100", "official 50k/10k split"),
    "svhn": DatasetSpec("svhn", 10, "torchvision.datasets.SVHN", "official 73257/26032 split"),
    "eurosat": DatasetSpec(
        "eurosat",
        10,
        "torchvision.datasets.EuroSAT",
        "no official split in torchvision: stratified 1620/540 per class",
    ),
    "dtd": DatasetSpec("dtd", 47, "torchvision.datasets.DTD", "official train(1880)/test(1880) split, partition 1"),
}

BENCHMARK_ORDER = ["cifar10", "cifar100", "svhn", "eurosat", "dtd"]


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------


def build_splits(name: str, raw_root: Path, *, seed: int = 20260921):
    """Return ``(train_dataset, test_dataset, class_names, meta)`` for one benchmark."""

    from torch.utils.data import Subset
    import torchvision.datasets as tvd

    root = Path(raw_root)
    root.mkdir(parents=True, exist_ok=True)
    spec = SPECS[name]
    meta: dict[str, Any] = {"benchmark": name, "source": spec.source, "split_note": spec.note}
    if name == "cifar10":
        train = tvd.CIFAR10(root, train=True, download=True)
        test = tvd.CIFAR10(root, train=False, download=True)
        classes = list(train.classes)
    elif name == "cifar100":
        train = tvd.CIFAR100(root, train=True, download=True)
        test = tvd.CIFAR100(root, train=False, download=True)
        classes = list(train.classes)
    elif name == "svhn":
        train = tvd.SVHN(root, split="train", download=True)
        test = tvd.SVHN(root, split="test", download=True)
        classes = [str(index) for index in range(10)]
    elif name == "eurosat":
        full = tvd.EuroSAT(root, download=True)
        classes = list(full.classes)
        labels = np.asarray([int(full[index][1]) for index in range(len(full))], dtype=np.int64)
        rng = np.random.default_rng(seed)
        train_index: list[int] = []
        test_index: list[int] = []
        for class_index in range(len(classes)):
            members = np.flatnonzero(labels == class_index)
            members = members[rng.permutation(members.size)]
            take_train = min(EUROSAT_TRAIN_PER_CLASS, members.size)
            take_test = min(EUROSAT_TEST_PER_CLASS, members.size - take_train)
            train_index.extend(members[:take_train].tolist())
            test_index.extend(members[take_train : take_train + take_test].tolist())
        train = Subset(full, sorted(train_index))
        test = Subset(full, sorted(test_index))
        meta["stratified_split_seed"] = int(seed)
    elif name == "dtd":
        train = tvd.DTD(root, split="train", download=True)
        test = tvd.DTD(root, split="test", download=True)
        classes = list(train.classes)
    else:
        raise ValueError(f"unknown benchmark: {name}")
    meta["classes"] = classes
    meta["num_classes"] = len(classes)
    meta["train_size"] = int(len(train))
    meta["test_size"] = int(len(test))
    return train, test, classes, meta


# --------------------------------------------------------------------------
# CLIP encoding
# --------------------------------------------------------------------------


def load_clip(device: str):
    """Frozen CLIP ViT-L-14 with the exact preprocessing used for the cache."""

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, cache_dir=CLIP_CACHE_DIR
    )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess


def encode_split(
    model,
    preprocess,
    dataset,
    *,
    device: str,
    batch_size: int = 128,
    num_workers: int = 6,
    log_every: int = 100,
    amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one split to L2-normalised float16 embeddings plus labels.

    ``amp`` runs the frozen encoder under CUDA autocast(float16).  The shared
    GPU of this workspace is heavily contended, and mixed precision is ~2.6x
    faster there; the cached float16 embeddings are the single source of truth
    for every later stage, so all methods see exactly the same features.
    """

    import torch
    from torch.utils.data import DataLoader

    wrapped = _Preprocessed(dataset, preprocess)
    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    embeddings = np.empty((len(dataset), EMBED_DIM), dtype=np.float16)
    labels = np.empty(len(dataset), dtype=np.int64)
    cursor = 0
    started = time.perf_counter()
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=bool(amp and str(device).startswith("cuda")))
    with torch.no_grad():
        for step, (images, targets) in enumerate(loader):
            with autocast:
                features = model.encode_image(images.to(device, non_blocking=True))
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rows = features.shape[0]
            embeddings[cursor : cursor + rows] = features.to(torch.float16).cpu().numpy()
            labels[cursor : cursor + rows] = targets.numpy().astype(np.int64)
            cursor += rows
            if log_every and (step + 1) % log_every == 0:
                rate = cursor / max(time.perf_counter() - started, 1e-6)
                print(f"    {cursor}/{len(dataset)} images ({rate:.1f} img/s)", flush=True)
    if cursor != len(dataset):
        raise RuntimeError(f"encoded {cursor} rows but dataset has {len(dataset)}")
    return embeddings, labels


class _Preprocessed:
    """Map-style wrapper applying the CLIP preprocessing to (image, label) pairs."""

    def __init__(self, dataset, preprocess) -> None:
        self.dataset = dataset
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return self.preprocess(image), int(label)


def features_dir(feature_root: Path, name: str) -> Path:
    return Path(feature_root) / name


def feature_paths(feature_root: Path, name: str) -> dict[str, Path]:
    directory = features_dir(feature_root, name)
    return {
        "train_emb": directory / "train_emb.npy",
        "train_labels": directory / "train_labels.npy",
        "test_emb": directory / "test_emb.npy",
        "test_labels": directory / "test_labels.npy",
        "meta": directory / "meta.json",
    }


def cache_is_complete(feature_root: Path, name: str) -> bool:
    paths = feature_paths(feature_root, name)
    return all(path.exists() for path in paths.values())


def extract_benchmark(
    name: str,
    *,
    raw_root: Path,
    feature_root: Path,
    device: str,
    batch_size: int = 128,
    num_workers: int = 6,
    seed: int = 20260921,
    delete_raw: bool = True,
    amp: bool = True,
) -> dict[str, Any]:
    """Extract and cache both splits of one benchmark; optionally delete raw files."""

    import shutil

    if cache_is_complete(feature_root, name):
        existing = json.loads(feature_paths(feature_root, name)["meta"].read_text(encoding="utf-8"))
        print(f"[{name}] cache already complete, skipping", flush=True)
        return existing

    model, preprocess = load_clip(device)
    train, test, classes, meta = build_splits(name, raw_root, seed=seed)
    directory = features_dir(feature_root, name)
    directory.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "benchmark": name,
        "clip_model": CLIP_MODEL,
        "clip_pretrained": CLIP_PRETRAINED,
        "clip_cache_dir": CLIP_CACHE_DIR,
        "embedding_dim": EMBED_DIM,
        "dtype": "float16",
        "normalised": "l2",
        "preprocess": "open_clip create_model_and_transforms val transform (resize 224, centre crop, CLIP mean/std)",
        "device": device,
        "batch_size": int(batch_size),
        "amp_autocast_fp16": bool(amp),
        "extracted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **meta,
    }
    for split, dataset in (("train", train), ("test", test)):
        started = time.perf_counter()
        print(f"[{name}] encoding {split} split ({len(dataset)} images)", flush=True)
        embeddings, labels = encode_split(
            model,
            preprocess,
            dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            amp=amp,
        )
        np.save(feature_paths(feature_root, name)[f"{split}_emb"], embeddings)
        np.save(feature_paths(feature_root, name)[f"{split}_labels"], labels)
        record[f"{split}_seconds"] = float(time.perf_counter() - started)
        record[f"{split}_size"] = int(len(dataset))
        record[f"{split}_label_counts"] = np.bincount(labels, minlength=len(classes)).astype(int).tolist()
    feature_paths(feature_root, name)["meta"].write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    del model
    import torch

    torch.cuda.empty_cache()
    if delete_raw:
        # Raw image files are no longer needed: embeddings are cached.
        _delete_raw_for(name, Path(raw_root), Path(feature_root), record)
        feature_paths(feature_root, name)["meta"].write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    print(f"[{name}] cached features in {directory}", flush=True)
    return record


def _delete_raw_for(name: str, raw_root: Path, feature_root: Path, record: dict[str, Any]) -> None:
    """Delete only the raw artefacts belonging to ``name`` (never other benchmarks)."""

    import shutil

    targets: list[Path] = []
    if name == "cifar10":
        targets = [raw_root / "cifar-10-python.tar.gz", raw_root / "cifar-10-batches-py"]
    elif name == "cifar100":
        targets = [raw_root / "cifar-100-python.tar.gz", raw_root / "cifar-100-python"]
    elif name == "svhn":
        targets = [raw_root / "train_32x32.mat", raw_root / "test_32x32.mat", raw_root / "extra_32x32.mat"]
    elif name == "eurosat":
        targets = [raw_root / "eurosat" / "EuroSAT_RGB.zip"]
        targets += [path for path in (raw_root / "eurosat").glob("*/") if path.is_dir()]
    elif name == "dtd":
        targets = [raw_root / "dtd" / "dtd-r1.0.1.tar.gz", raw_root / "dtd" / "dtd"]
    removed = []
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            removed.append(str(target))
        elif target.exists():
            target.unlink()
            removed.append(str(target))
    record["raw_deleted"] = removed


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Features:
    benchmark: str
    train_emb: np.ndarray
    train_labels: np.ndarray
    test_emb: np.ndarray
    test_labels: np.ndarray
    class_names: list[str]
    meta: dict[str, Any]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def embed_dim(self) -> int:
        return int(self.train_emb.shape[1])


def load_features(feature_root: Path, name: str) -> Features:
    paths = feature_paths(feature_root, name)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing cached features for {name}: {missing}")
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    return Features(
        benchmark=name,
        train_emb=np.load(paths["train_emb"], mmap_mode="r"),
        train_labels=np.load(paths["train_labels"], mmap_mode="r"),
        test_emb=np.load(paths["test_emb"], mmap_mode="r"),
        test_labels=np.load(paths["test_labels"], mmap_mode="r"),
        class_names=list(meta["classes"]),
        meta=meta,
    )
