# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from cached_path import cached_path

MODEL_CONFIGS = {
    "12hz_v1": {
        "fname": "dualcodec_12hz_16384_4096.safetensors",
        "cfgname": "dualcodec_12hz_16384_4096_8vq.yaml",
        "repo": "hf://amphion/dualcodec",
        "semantic_model_type": "w2vbert",
        "max_quantizers": 8,
        "skip_semantic_normalize": False,
    },
    "25hz_v1": {
        "fname": "dualcodec_25hz_16384_1024.safetensors",
        "cfgname": "dualcodec_25hz_16384_1024_12vq.yaml",
        "repo": "hf://amphion/dualcodec",
        "semantic_model_type": "w2vbert",
        "max_quantizers": 12,
        "skip_semantic_normalize": False,
    },
    "12hz_v1.5_sensevoice": {
        "fname": "dualcodec_12hz_v1.5_sensevoice.safetensors",
        "cfgname": "dualcodec_12hz_sensevoice.yaml",
        "repo": "hf://jiaqili3/dualcodec12hz_v1.5_sensevoice",
        "semantic_model_type": "sensevoice",
        "max_quantizers": 8,
        "skip_semantic_normalize": True,
    },
}

model_id_to_fname = {k: v["fname"] for k, v in MODEL_CONFIGS.items()}
model_id_to_cfgname = {k: v["cfgname"] for k, v in MODEL_CONFIGS.items()}
model_id_to_repo = {k: v["repo"] for k, v in MODEL_CONFIGS.items()}


def get_model(model_id="12hz_v1", pretrained_model_path=None):
    """Load a DualCodec model by Model_ID.

    If ``pretrained_model_path`` is omitted, weights are downloaded from the
    Hugging Face repo registered for that Model_ID.
    """
    import os

    if model_id not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_id={model_id!r}. Available: {list(MODEL_CONFIGS)}"
        )
    model_cfg = MODEL_CONFIGS[model_id]
    if pretrained_model_path is None:
        pretrained_model_path = model_cfg["repo"]

    pretrained_model_path = cached_path(pretrained_model_path)

    import hydra
    from hydra import initialize

    with initialize(version_base="1.3", config_path="../../conf/model"):
        cfg = hydra.compose(config_name=model_cfg["cfgname"], overrides=[])
        model = hydra.utils.instantiate(cfg.model)

    if pretrained_model_path is None:
        import warnings

        warnings.warn(
            "pretrained_model_path is not given, model will be loaded without weights"
        )
    else:
        model_fname = os.path.join(pretrained_model_path, model_cfg["fname"])
        if not os.path.isfile(model_fname):
            raise FileNotFoundError(
                f"Checkpoint not found: {model_fname}. "
                f"Pass a local directory that contains {model_cfg['fname']}, "
                f"or omit pretrained_model_path to download from {model_cfg['repo']}."
            )
        print("Loading model from", model_fname)
        import safetensors.torch

        safetensors.torch.load_model(model, model_fname)
        print("Model loaded")
    model.eval()
    model.model_id = model_id
    model.semantic_model_type = model_cfg["semantic_model_type"]
    model.skip_semantic_normalize = model_cfg["skip_semantic_normalize"]
    return model
