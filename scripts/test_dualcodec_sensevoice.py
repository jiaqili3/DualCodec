#!/usr/bin/env python
"""Smoke-test DualCodec-SenseVoice data + model forward before full training."""

from __future__ import annotations

import os
import sys

import torch


def main():
    python = sys.executable
    print(f"python={python}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from dualcodec.dataset.webdataset_audio import WebDatasetAudioShards
    from dualcodec.dataset.processor import (
        _build_fbank_feature_extractor,
        _build_sensevoice_semantic_model,
        fbank_feature,
        gluster_filter,
        gluster_opener,
        gluster_padding,
        resample,
        segment_speech,
        batch,
    )
    from dualcodec.model_codec.dualcodec_model import DualCodec

    shards = WebDatasetAudioShards(
        shard_patterns=[
            "/F00120260003/flexislm_project/data/S2TT/emilia_zh-en/shards/train-00000-of-00096.tar",
            "/F00120260003/flexislm_project/data/S2TT/common_voice/shards/train-00000-of-00111.tar",
        ],
        shuffle_shards=False,
        partition_by_rank=False,
        min_seconds=0.5,
        max_seconds=40.0,
    )

    def _as_datalist(src_iter):
        for item in src_iter:
            yield {"src": item}

    pipe = _as_datalist(iter(shards))
    pipe = gluster_opener(pipe, manual_dist_sampler=False, min_seconds=0.5, max_seconds=40.0)
    pipe = gluster_filter(pipe, is_emilia=True, ignore_text=True)
    pipe = resample(pipe, resample_rate=24000)
    pipe = segment_speech(pipe, segment_length=24000)
    feat_extractor = _build_fbank_feature_extractor(sr=16000)
    pipe = fbank_feature(pipe, feature_extractor=feat_extractor)
    pipe = batch(pipe, batch_type="static", batch_size=2, ignore_text=True)
    pipe = gluster_padding(pipe, return_speech=True, ignore_text=True)

    batch_data = next(iter(pipe))
    print(
        "batch keys:",
        sorted(batch_data.keys()),
        "speech",
        tuple(batch_data["speech"].shape),
        "feats",
        tuple(batch_data["input_features"].shape),
        "mask",
        tuple(batch_data["attention_mask"].shape),
    )
    assert batch_data["input_features"].shape[-1] == 560, batch_data["input_features"].shape

    sensevoice = _build_sensevoice_semantic_model(
        sensevoice_model_path="/F00120260003/flexislm_project/model/SenseVoiceSmall",
        sensevoice_prepend_inputs=True,
    )
    semantic_model = sensevoice["model"].to(device)

    input_features = batch_data["input_features"].to(device)
    attention_mask = batch_data["attention_mask"].to(device)
    lengths = attention_mask.sum(dim=-1).long()
    feats_in, lengths = semantic_model.prepend_inputs(input_features.float(), lengths)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
        _o, _ol, hidden_out, _h = semantic_model.encoder(
            feats_in, lengths, extract_hidden=True
        )
    feat = hidden_out[:, 4:].transpose(1, 2)  # [B, 512, T] @ ~16.67Hz
    print("sensevoice feat", tuple(feat.shape), "expected C=512")
    assert feat.shape[1] == 512

    factor = 1.33333
    target_length = int(feat.shape[-1] / factor)
    feat_12hz = torch.nn.functional.interpolate(
        feat, size=target_length, mode="linear", align_corners=False
    )
    print("interpolated 12.5Hz feat", tuple(feat_12hz.shape))

    model = DualCodec(
        sample_rate=24000,
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
        semantic_downsample_factor=1.33333,
    ).to(device)
    model.train()

    speech = batch_data["speech"].float()[:, None, :].to(device)
    # Align semantic frames to DAC hop if off by a few frames
    dac_frames = speech.shape[-1] // 1920
    if feat_12hz.shape[-1] != dac_frames:
        feat_12hz = torch.nn.functional.interpolate(
            feat_12hz, size=dac_frames, mode="linear", align_corners=False
        )
        print("re-aligned semantic to dac frames", dac_frames, tuple(feat_12hz.shape))

    out_dict, semantic_edict = model(
        speech,
        semantic_repr=feat_12hz,
        bypass_quantize_rate=0.0,
        possibly_no_quantizer=False,
    )
    print(
        "forward ok: recon",
        tuple(out_dict.x.shape),
        "semantic",
        tuple(semantic_edict["x"].shape),
        "commitment",
        float(out_dict.penalty.detach().cpu()),
    )
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
