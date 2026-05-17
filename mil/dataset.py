# -*- coding: utf-8 -*-
"""Side-aware bag dataset for SS-AttnMIL."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import zlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import SEED


def read_bags_by_side(meta_path: str) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, int]]:
    """Read patch metadata and group patches by patient and side."""
    df = pd.read_csv(meta_path)
    required = {"ID", "Cat", "patch_path", "side"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")
    df["ID"] = df["ID"].astype(str)
    df["Cat"] = df["Cat"].astype(int)
    df["side"] = df["side"].str.lower().str.strip()

    bags = {}
    for pid, group in df.groupby("ID"):
        sides = {}
        for side_name, side_group in group.groupby("side"):
            sides[side_name] = side_group["patch_path"].astype(str).tolist()
        bags[pid] = sides
    labels = df.groupby("ID")["Cat"].first().to_dict()
    return bags, labels


def _load_patches(paths: List[str], max_inst: int, rng: np.random.RandomState) -> np.ndarray:
    """Load and normalize sampled .npy CT patches."""
    if len(paths) > max_inst:
        indices = rng.choice(len(paths), max_inst, replace=False)
        paths = [paths[i] for i in indices]

    patches = []
    for patch_path in paths:
        path = Path(patch_path)
        if not path.exists():
            raise FileNotFoundError(f"Patch file listed in metadata does not exist: {path}")
        try:
            arr = np.load(path).astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Failed to load patch file: {path}") from exc
        arr = np.clip((arr + 200.0) / 1000.0, 0.0, 1.0)
        patches.append(arr[None])
    if not patches:
        raise ValueError("Cannot load an empty patch list.")
    return np.stack(patches, axis=0)


class SideAwareBagDataset(Dataset):
    """Return left/right sub-bags for each patient."""

    def __init__(
        self,
        ids: List[str],
        bags: Dict[str, Dict[str, List[str]]],
        labels: Dict[str, int],
        max_inst_per_side: int = 24,
        patch_shape: Tuple[int, ...] = (96, 96, 32),
        seed: int = SEED,
        deterministic_sampling: bool = False,
    ):
        self.ids = ids
        self.bags = bags
        self.labels = labels
        self.max_inst_per_side = max_inst_per_side
        self.patch_shape = patch_shape
        self.seed = seed
        self.deterministic_sampling = deterministic_sampling
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def _rng_for_patient(self, pid: str) -> np.random.RandomState:
        if not self.deterministic_sampling:
            return self.rng
        local_seed = self.seed + zlib.crc32(str(pid).encode("utf-8"))
        return np.random.RandomState(local_seed % (2**32 - 1))

    def __getitem__(self, idx):
        pid = self.ids[idx]
        sides = self.bags[pid]
        y = int(self.labels[pid])
        rng = self._rng_for_patient(pid)

        if "left" in sides and sides["left"]:
            left_patches = _load_patches(sides["left"], self.max_inst_per_side, rng)
            left_valid = 1.0
        else:
            left_patches = np.zeros((1, 1, *self.patch_shape), dtype=np.float32)
            left_valid = 0.0

        if "right" in sides and sides["right"]:
            right_patches = _load_patches(sides["right"], self.max_inst_per_side, rng)
            right_valid = 1.0
        else:
            right_patches = np.zeros((1, 1, *self.patch_shape), dtype=np.float32)
            right_valid = 0.0

        return (
            pid,
            torch.from_numpy(left_patches),
            torch.from_numpy(right_patches),
            torch.tensor([left_valid, right_valid], dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )


def collate_side_bags(batch):
    """Keep variable-length side bags as Python lists."""
    pids = [b[0] for b in batch]
    left_list = [b[1] for b in batch]
    right_list = [b[2] for b in batch]
    valid_mask = torch.stack([b[3] for b in batch], dim=0)
    ys = torch.stack([b[4] for b in batch], dim=0)
    return pids, left_list, right_list, valid_mask, ys
