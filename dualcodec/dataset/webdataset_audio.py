# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Streaming webdataset shards that yield Emilia-compatible audio samples."""

from __future__ import annotations

import glob
import io
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from torch.utils.data import IterableDataset


AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus")
AUDIO_EXT_NAMES = tuple(suf.lstrip(".") for suf in AUDIO_SUFFIXES)


def expand_shard_patterns(patterns: Union[str, Sequence[str]]) -> List[str]:
    """Expand brace/glob shard patterns into sorted unique tar paths."""
    if isinstance(patterns, str):
        patterns = [patterns]
    shards: List[str] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(os.path.expanduser(pattern)))
        if not expanded and os.path.isfile(pattern):
            expanded = [pattern]
        if not expanded:
            raise FileNotFoundError(f"No shards matched pattern: {pattern}")
        shards.extend(expanded)
    seen = set()
    unique = []
    for path in shards:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _pick_audio_key(sample: dict) -> Optional[str]:
    keys = []
    for key in sample.keys():
        lower = key.lower()
        if any(lower.endswith(suf) for suf in AUDIO_SUFFIXES):
            keys.append(key)
        elif lower in AUDIO_EXT_NAMES:
            keys.append(key)
    if not keys:
        return None
    for key in keys:
        lower = key.lower()
        if lower == "audio.mp3" or lower.endswith(".audio.mp3") or "audio." in lower:
            return key
    return keys[0]


def _decode_audio_bytes(data: bytes, codec: Optional[str] = None):
    """Decode in-tar audio bytes without torchcodec (broken on some nodes)."""
    buffer = io.BytesIO(data)
    codec = (codec or "").lower()

    # Prefer soundfile / librosa; torchaudio 2.10+ routes mp3 through torchcodec.
    try:
        import soundfile as sf

        array, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)
        waveform = torch.as_tensor(array, dtype=torch.float32)
        if waveform.dim() > 1:
            waveform = waveform[:, 0]
        return waveform.contiguous(), int(sample_rate)
    except Exception:
        buffer.seek(0)

    try:
        import librosa

        array, sample_rate = librosa.load(buffer, sr=None, mono=True)
        waveform = torch.as_tensor(array, dtype=torch.float32)
        return waveform.contiguous(), int(sample_rate)
    except Exception:
        buffer.seek(0)

    # Last resort: torchaudio (may require working ffmpeg/torchcodec).
    if codec:
        buffer.name = f"audio.{codec}"
    waveform, sample_rate = torchaudio.load(buffer)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform[:1]
    return waveform.squeeze(0).contiguous(), int(sample_rate)


class WebDatasetAudioShards(IterableDataset):
    """Iterate tar shards and yield samples for ``gluster_filter(is_emilia=True)``.

    Each yielded example looks like::

        {
            "mp3": {"array": np.ndarray[T], "sampling_rate": int},
            "duration": float,
            "key": str,
            "__url__": str,
        }
    """

    def __init__(
        self,
        shard_patterns: Union[str, Sequence[str]],
        shuffle_shards: bool = True,
        shard_shuffle_seed: int = 42,
        min_seconds: float = 0.5,
        max_seconds: float = 45.0,
        max_shards: Optional[int] = None,
        partition_by_rank: bool = True,
    ):
        super().__init__()
        self.shards = expand_shard_patterns(shard_patterns)
        if max_shards is not None:
            self.shards = self.shards[: int(max_shards)]
        if not self.shards:
            raise ValueError("WebDatasetAudioShards received an empty shard list")
        self.shuffle_shards = shuffle_shards
        self.shard_shuffle_seed = shard_shuffle_seed
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.partition_by_rank = partition_by_rank

        # Snapshot rank-local shards in __init__: dataloader workers usually do
        # not have torch.distributed initialized.
        shards = list(self.shards)
        if self.shuffle_shards:
            rng = np.random.default_rng(self.shard_shuffle_seed)
            rng.shuffle(shards)
        self.rank = 0
        self.world_size = 1
        if self.partition_by_rank and dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            shards = shards[self.rank :: self.world_size]
        self.rank_shards = shards
        print(
            f"[WebDatasetAudioShards] total={len(self.shards)} "
            f"rank{self.rank}/{self.world_size} local={len(self.rank_shards)} "
            f"(min={min_seconds}s, max={max_seconds}s)"
        )

    def __len__(self):
        return len(self.rank_shards)

    def _local_shards(self) -> List[str]:
        shards = list(self.rank_shards)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]
        return shards

    def __iter__(self):
        try:
            import webdataset as wds
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "webdataset is required for WebDatasetAudioShards"
            ) from exc

        # Avoid double-splitting: we already sliced by rank/worker above.
        identity = lambda src: src
        for shard in self._local_shards():
            raw = wds.WebDataset(
                shard,
                shardshuffle=False,
                nodesplitter=identity,
                workersplitter=identity,
                handler=wds.warn_and_continue,
                empty_check=False,
            )
            for sample in raw:
                try:
                    audio_key = _pick_audio_key(sample)
                    if audio_key is None:
                        continue
                    payload = sample[audio_key]
                    codec = audio_key.rsplit(".", 1)[-1].lower()
                    if isinstance(payload, bytes):
                        waveform, sr = _decode_audio_bytes(payload, codec=codec)
                    elif isinstance(payload, dict) and "array" in payload:
                        waveform = torch.as_tensor(
                            payload["array"], dtype=torch.float32
                        )
                        sr = int(payload.get("sampling_rate", 16000))
                        if waveform.dim() > 1:
                            waveform = waveform.reshape(-1)
                    else:
                        continue
                    duration = float(waveform.numel()) / float(sr)
                    if duration < self.min_seconds or duration > self.max_seconds:
                        continue
                    yield {
                        "mp3": {
                            "array": waveform.numpy().astype(np.float32, copy=False),
                            "sampling_rate": sr,
                        },
                        "duration": duration,
                        "key": sample.get("__key__", ""),
                        "__url__": sample.get("__url__", shard),
                    }
                except Exception as exc:
                    # Avoid dumping huge torchcodec/ffmpeg traces per sample.
                    msg = str(exc).splitlines()[0][:200]
                    print(
                        f"[WebDatasetAudioShards] skip sample in {shard}: {msg}"
                    )
                    continue
