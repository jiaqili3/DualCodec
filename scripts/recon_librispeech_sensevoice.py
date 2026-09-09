#!/usr/bin/env python
"""Reconstruct LibriSpeech test-clean with DualCodec-SenseVoice at 1 VQ and 8 VQ."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


ROOT = Path("/F00120260003/flexislm_project/jiaqi/DualCodec")
DEFAULT_TEST_CLEAN = Path(
    "/F00120260003/flexislm_project/data/eval/LibriSpeech/librispeech/LibriSpeech/test-clean"
)
DEFAULT_CKPT_ROOT = (
    ROOT / "output_checkpoints" / "dualcodec_12hz_sensevoice" / "checkpoint"
)
SENSEVOICE_PATH = "/F00120260003/flexislm_project/model/SenseVoiceSmall"
SAMPLE_RATE = 24000
HOP = 4 * 5 * 6 * 8 * 2  # 1920
SEMANTIC_DOWNSAMPLE = 1.33333


def latest_checkpoint(ckpt_root: Path) -> Path:
    ckpts = [
        p
        for p in ckpt_root.iterdir()
        if p.is_dir() and p.name.startswith("epoch")
    ]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {ckpt_root}")

    def step_of(path: Path) -> int:
        return int(path.name.split("_")[1].split("-")[1])

    return max(ckpts, key=step_of)


def load_codec(ckpt_dir: Path, device: torch.device):
    from dualcodec.model_codec.dualcodec_model import DualCodec
    import safetensors.torch

    model = DualCodec(
        sample_rate=SAMPLE_RATE,
        encoder_rates=[4, 5, 6, 8, 2],
        decoder_rates=[2, 8, 6, 5, 4],
        encoder_dim=32,
        decoder_dim=1536,
        latent_dim=512,
        ssl_dim=512,
        n_codebooks=7,
        quantizer_dropout=1.0,
        codebook_size=4096,
        semantic_codebook_size=16384,
        is_causal=True,
        semantic_downsample_factor=SEMANTIC_DOWNSAMPLE,
    )
    weight_path = ckpt_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing generator weights: {weight_path}")
    missing, unexpected = safetensors.torch.load_model(model, str(weight_path))
    if missing or unexpected:
        print(
            f"[warn] load_model missing={missing} unexpected={unexpected}",
            flush=True,
        )
    model.eval().to(device)
    return model


def load_sensevoice(device: torch.device):
    from dualcodec.dataset.processor import (
        _build_fbank_feature_extractor,
        _build_sensevoice_semantic_model,
    )

    feat_extractor = _build_fbank_feature_extractor(sr=16000)
    cfg = _build_sensevoice_semantic_model(
        sensevoice_model_path=SENSEVOICE_PATH,
        sensevoice_prepend_inputs=True,
    )
    semantic_model = cfg["model"].to(device).eval()
    return feat_extractor, semantic_model, cfg["sensevoice_prepend_inputs"]


@torch.no_grad()
def extract_semantic(
    wav_24k: torch.Tensor,
    feat_extractor,
    semantic_model,
    prepend_inputs: bool,
    device: torch.device,
) -> torch.Tensor:
    wav_16k = torchaudio.functional.resample(wav_24k, SAMPLE_RATE, 16000)
    if wav_16k.dim() == 2:
        wav_16k = wav_16k.squeeze(0)
    mel, _ = feat_extractor.extract_fbank(wav_16k)
    if mel.dim() == 3:
        mel = mel.squeeze(0)
    input_features = mel.unsqueeze(0).to(device)
    lengths = torch.tensor([input_features.shape[1]], device=device, dtype=torch.long)
    if prepend_inputs:
        input_features, lengths = semantic_model.prepend_inputs(
            input_features.float(), lengths
        )
    with torch.amp.autocast(device_type=device.type, enabled=False):
        _o, _ol, hidden_out, _h = semantic_model.encoder(
            input_features.float(),
            lengths,
            extract_hidden=True,
        )
    feat = hidden_out[:, 4:].transpose(1, 2)  # [1, 512, T] @ ~16.67Hz
    target_length = int(feat.shape[-1] / SEMANTIC_DOWNSAMPLE)
    if target_length < 1:
        target_length = 1
    feat = torch.nn.functional.interpolate(
        feat, size=target_length, mode="linear", align_corners=False
    )
    dac_frames = max(1, wav_24k.shape[-1] // HOP)
    if feat.shape[-1] != dac_frames:
        feat = torch.nn.functional.interpolate(
            feat, size=dac_frames, mode="linear", align_corners=False
        )
    return feat


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load mono audio without torchcodec (broken on these nodes)."""
    try:
        import soundfile as sf

        array, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(array[:, 0].copy())
        return wav.unsqueeze(0), int(sr)
    except Exception:
        import librosa

        array, sr = librosa.load(str(path), sr=None, mono=True)
        wav = torch.from_numpy(np.asarray(array, dtype=np.float32))
        return wav.unsqueeze(0), int(sr)


def save_wav(path: Path, audio: torch.Tensor, orig_len: int) -> None:
    audio = audio.detach().float().cpu()
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[-1] < orig_len:
        audio = torch.nn.functional.pad(audio, (0, orig_len - audio.shape[-1]))
    else:
        audio = audio[..., :orig_len]
    peak = audio.abs().max().clamp_min(1e-8)
    if peak > 1.0:
        audio = audio / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(str(path), audio.squeeze(0).numpy(), SAMPLE_RATE, subtype="PCM_16")


def iter_audio_files(root: Path):
    files = sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in {".flac", ".wav", ".mp3", ".ogg"}
    )
    return files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--test-clean", type=Path, default=DEFAULT_TEST_CLEAN)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-quantizers", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    ckpt_dir = args.ckpt_dir or latest_checkpoint(DEFAULT_CKPT_ROOT)
    ckpt_dir = ckpt_dir.resolve()
    step = int(ckpt_dir.name.split("_")[1].split("-")[1])
    out_root = args.out_dir or (
        ROOT
        / "output_checkpoints"
        / "dualcodec_12hz_sensevoice"
        / "eval_recon"
        / f"step-{step:07d}"
        / "librispeech-test-clean"
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[recon] ckpt={ckpt_dir}", flush=True)
    print(f"[recon] device={device} cuda={torch.cuda.is_available()}", flush=True)
    print(f"[recon] out={out_root}", flush=True)

    files = iter_audio_files(args.test_clean)
    if args.limit is not None:
        files = files[: args.limit]
    print(f"[recon] files={len(files)} nq={args.n_quantizers}", flush=True)
    if not files:
        raise FileNotFoundError(f"No audio under {args.test_clean}")

    codec = load_codec(ckpt_dir, device)
    feat_extractor, semantic_model, prepend = load_sensevoice(device)

    done = 0
    skipped = 0
    failed = 0
    for path in tqdm(files, desc="reconstruct"):
        rel = path.relative_to(args.test_clean).with_suffix(".wav")
        targets = {nq: out_root / f"{nq}vq" / rel for nq in args.n_quantizers}
        if args.skip_existing and all(p.is_file() for p in targets.values()):
            skipped += 1
            continue
        try:
            wav, sr = load_audio(path)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            orig_len = wav.shape[-1]
            if orig_len < HOP:
                wav = torch.nn.functional.pad(wav, (0, HOP - orig_len))
            wav = wav.to(device)
            feat = extract_semantic(
                wav, feat_extractor, semantic_model, prepend, device
            )
            audio_in = wav.unsqueeze(0) if wav.dim() == 2 else wav
            if audio_in.dim() == 2:
                audio_in = audio_in.unsqueeze(0)
            for nq, out_path in targets.items():
                if args.skip_existing and out_path.is_file():
                    continue
                semantic_codes, acoustic_codes = codec.encode(
                    audio_in, num_quantizers=nq, semantic_repr=feat
                )
                recon = codec.decode_from_codes(semantic_codes, acoustic_codes)
                save_wav(out_path, recon, orig_len)
            done += 1
        except Exception as exc:
            failed += 1
            print(f"[recon] fail {path}: {exc}", flush=True)

    print(
        f"[recon] done={done} skipped={skipped} failed={failed} total={len(files)}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
