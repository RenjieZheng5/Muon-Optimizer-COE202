from __future__ import annotations

import gzip
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class DatasetInfo:
    input_shape: tuple[int, int, int]
    num_classes: int


class FakeClassificationDataset(Dataset):
    def __init__(
        self,
        size: int,
        input_shape: tuple[int, int, int] = (1, 28, 28),
        num_classes: int = 10,
        seed: int = 0,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(size, *input_shape, generator=generator)
        flat = self.x.flatten(1)
        signal = flat[:, :num_classes]
        self.y = signal.argmax(dim=1) % num_classes

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def _maybe_subset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


DATASET_URLS = {
    "mnist": {
        "base": "https://ossci-datasets.s3.amazonaws.com/mnist",
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    },
    "fashion_mnist": {
        "base": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion",
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    },
}


def _download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


def _read_idx_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2051:
        raise ValueError(f"Invalid image IDX file {path}: magic={magic}")
    count = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    array = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols)
    return torch.from_numpy(array.copy()).float().unsqueeze(1).div_(255.0)


def _read_idx_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2049:
        raise ValueError(f"Invalid label IDX file {path}: magic={magic}")
    count = int.from_bytes(data[4:8], "big")
    array = np.frombuffer(data, dtype=np.uint8, offset=8)
    if array.shape[0] != count:
        raise ValueError(f"Invalid label count in {path}: expected={count}, got={array.shape[0]}")
    return torch.from_numpy(array.copy()).long()


def _load_idx_dataset(name: str, data_dir: str, train: bool) -> TensorImageDataset:
    spec = DATASET_URLS[name]
    split = "train" if train else "test"
    image_key = f"{split}_images"
    label_key = f"{split}_labels"
    root = Path(data_dir) / name
    image_path = root / spec[image_key]
    label_path = root / spec[label_key]
    _download_if_missing(f"{spec['base']}/{spec[image_key]}", image_path)
    _download_if_missing(f"{spec['base']}/{spec[label_key]}", label_path)
    return TensorImageDataset(_read_idx_images(image_path), _read_idx_labels(label_path))


def build_loaders(
    dataset_name: str,
    batch_size: int,
    data_dir: str = "data",
    seed: int = 0,
    subset_train: Optional[int] = None,
    subset_val: Optional[int] = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DatasetInfo]:
    generator = torch.Generator().manual_seed(seed)

    if dataset_name == "fake":
        train_set = FakeClassificationDataset(subset_train or 1024, seed=seed)
        val_set = FakeClassificationDataset(subset_val or 256, seed=seed + 1000)
        info = DatasetInfo(input_shape=(1, 28, 28), num_classes=10)
    elif dataset_name in {"mnist", "fashion_mnist"}:
        train_set = _load_idx_dataset(dataset_name, data_dir, train=True)
        test_set = _load_idx_dataset(dataset_name, data_dir, train=False)
        train_set = _maybe_subset(train_set, subset_train)
        val_set = _maybe_subset(test_set, subset_val)
        info = DatasetInfo(input_shape=(1, 28, 28), num_classes=10)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, info
