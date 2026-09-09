# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torchaudio
import torch.nn.functional as F
import os
from easydict import EasyDict as edict
from contextlib import nullcontext
import warnings


def _build_semantic_model(
    dualcodec_path,
    meanvar_fname="w2vbert2_mean_var_stats_emilia.pt",
    semantic_model_path="facebook/w2v-bert-2.0",
    device="cuda",
    **kwargs,
):
    """Build the w2v semantic model and load pretrained weights.
    Inputs:
    - dualcodec_path: str, path to the dualcodec model
    - meanvar_fname: str, filename of the mean and variance statistics
    - semantic_model_path: str, path to the semantic model, or a huggngface model name
    Outputs:
    cfg: edict, containing the semantic model, mean, std, and feature extractor.
    - model: Wav2Vec2BertModel instance for semantic feature extraction
    - layer_idx: int (15), index of the layer to extract features from
    - output_idx: int (17), index of the output layer (layer_idx + 2)
    - mean: torch.Tensor containing precomputed mean for feature normalization
    - std: torch.Tensor containing precomputed standard deviation for normalization
    - feature_extractor: SeamlessM4TFeatureExtractor for audio preprocessing
    """
    from transformers import Wav2Vec2BertModel

    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available, running on CPU.")
        device = "cpu"

    # load semantic model
    semantic_model = Wav2Vec2BertModel.from_pretrained(semantic_model_path)
    semantic_model = semantic_model.eval().to(device)

    # load feature extractor
    from transformers import SeamlessM4TFeatureExtractor

    w2v_feat_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        semantic_model_path
    )

    layer_idx = 15
    output_idx = layer_idx + 2

    # load mean and std
    meanvar_path = os.path.join(dualcodec_path, meanvar_fname)
    stat_mean_var = torch.load(meanvar_path)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean
    semantic_std = semantic_std

    return edict(
        {
            "semantic_model": semantic_model,
            "layer_idx": layer_idx,
            "output_idx": output_idx,
            "mean": semantic_mean,
            "std": semantic_std,
            "feature_extractor": w2v_feat_extractor,
        }
    )


from cached_path import cached_path

_CLUSTER_SENSEVOICE = "/F00120260003/flexislm_project/model/SenseVoiceSmall"
_DEFAULT_SENSEVOICE_HF = "FunAudioLLM/SenseVoiceSmall"


def _resolve_sensevoice_path(sensevoice_path=None):
    """Prefer an explicit / env / local SenseVoice checkpoint; else Hugging Face."""
    if sensevoice_path:
        return sensevoice_path
    env_path = os.environ.get("DUALCODEC_SENSEVOICE_PATH")
    if env_path:
        return env_path
    if os.path.isdir(_CLUSTER_SENSEVOICE):
        return _CLUSTER_SENSEVOICE
    return _DEFAULT_SENSEVOICE_HF


def _build_sensevoice_inference_cfg(sensevoice_path, device="cuda"):
    from dualcodec.dataset.processor import (
        _build_fbank_feature_extractor,
        _build_sensevoice_semantic_model,
    )

    feat_extractor = _build_fbank_feature_extractor(sr=16000)
    cfg = _build_sensevoice_semantic_model(
        sensevoice_model_path=sensevoice_path,
        sensevoice_prepend_inputs=True,
    )
    semantic_model = cfg["model"].to(device).eval()
    return edict(
        {
            "semantic_model": semantic_model,
            "feature_extractor": feat_extractor,
            "sensevoice_prepend_inputs": cfg.get("sensevoice_prepend_inputs", True),
            "skip_semantic_normalize": True,
        }
    )


class Inference:
    """
    Inference class for DualCodec.
    """

    def __init__(
        self,
        dualcodec_model,
        dualcodec_path=None,
        w2v_path=None,
        sensevoice_path=None,
        device="cuda",
        autocast=True,
        **kwargs,
    ) -> None:
        """
        Inputs:
        - dualcodec_model: DualCodec instance, the model weight is loaded by safetensors
        - dualcodec_path: str, path to the dualcodec model (w2v mean/var stats). Optional;
          defaults to hf://amphion/dualcodec for w2v-bert models.
        - w2v_path: str, path to the w2v-bert model. Optional; defaults to
          hf://facebook/w2v-bert-2.0. Unused for SenseVoice models.
        - sensevoice_path: str, local FunASR SenseVoiceSmall dir or HF/ModelScope id.
          Optional; auto-resolved for 12hz_v1.5_sensevoice.
        - device: str, device to run the model
        - autocast: bool, whether to use autocast to fp16 for model inference
        """
        if not torch.cuda.is_available():
            warnings.warn("CUDA is not available, running on CPU.")
            device = "cpu"

        self.semantic_model_type = getattr(
            dualcodec_model, "semantic_model_type", "w2vbert"
        )
        self.model = dualcodec_model
        self.model.to(device)
        self.model.eval()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.autocast = autocast

        if self.semantic_model_type == "sensevoice":
            resolved_sv = _resolve_sensevoice_path(sensevoice_path)
            print("Loading SenseVoice teacher from", resolved_sv)
            self.semantic_cfg = _build_sensevoice_inference_cfg(
                resolved_sv, device=str(self.device)
            )
        else:
            if dualcodec_path is None:
                dualcodec_path = "hf://amphion/dualcodec"
            if w2v_path is None:
                w2v_path = "hf://facebook/w2v-bert-2.0"
            dualcodec_path = cached_path(dualcodec_path)
            w2v_path = cached_path(w2v_path)
            self.semantic_cfg = _build_semantic_model(
                dualcodec_path=dualcodec_path,
                semantic_model_path=w2v_path,
                device=str(self.device),
                **kwargs,
            )

        for key in self.semantic_cfg:
            if isinstance(self.semantic_cfg[key], torch.nn.Module) or isinstance(
                self.semantic_cfg[key], torch.Tensor
            ):
                self.semantic_cfg[key] = self.semantic_cfg[key].to(self.device)

    def _autocast_ctx(self):
        if not self.autocast:
            return nullcontext()
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        if device_type != "cuda":
            return nullcontext()
        return torch.autocast(device_type=device_type, dtype=torch.float16)

    @torch.no_grad()
    def _extract_sensevoice_semantic_repr(self, audio):
        """SenseVoice FBank + encoder, interpolated to DualCodec 12.5Hz frames.

        Args:
        - audio: torch.Tensor, shape=(B, 1, T) at 24kHz
        Returns:
        - feat: torch.Tensor, shape=(B, 512, T_codec)
        """
        hop = int(getattr(self.model.dac, "hop_length", 1920))
        factor = float(self.model.semantic_downsample_factor)
        semantic_model = self.semantic_cfg.semantic_model
        feat_extractor = self.semantic_cfg.feature_extractor
        prepend = self.semantic_cfg.get("sensevoice_prepend_inputs", True)
        feats = []
        for i in range(audio.shape[0]):
            wav_24k = audio[i]
            if wav_24k.dim() == 2:
                wav_24k = wav_24k.squeeze(0)
            wav_16k = torchaudio.functional.resample(wav_24k.cpu(), 24000, 16000)
            mel, _ = feat_extractor.extract_fbank(wav_16k)
            if mel.dim() == 3:
                mel = mel.squeeze(0)
            input_features = mel.unsqueeze(0).to(self.device)
            lengths = torch.tensor(
                [input_features.shape[1]], device=self.device, dtype=torch.long
            )
            if prepend:
                input_features, lengths = semantic_model.prepend_inputs(
                    input_features.float(), lengths
                )
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                _o, _ol, hidden_out, _h = semantic_model.encoder(
                    input_features.float(),
                    lengths,
                    extract_hidden=True,
                )
            feat = hidden_out[:, 4:].transpose(1, 2)
            target_length = max(1, int(feat.shape[-1] / factor))
            feat = torch.nn.functional.interpolate(
                feat, size=target_length, mode="linear", align_corners=False
            )
            dac_frames = max(1, int(wav_24k.shape[-1] // hop))
            if feat.shape[-1] != dac_frames:
                feat = torch.nn.functional.interpolate(
                    feat, size=dac_frames, mode="linear", align_corners=False
                )
            feats.append(feat)
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode(
        self,
        audio,
        n_quantizers=8,
    ):
        """
        Args:
        - audio: torch.Tensor, shape=(B, 1, T), dtype=torch.float32, input audio waveform
        - n_quantizers: int, number of RVQ quantizers to use
        Returns:
        - semantic_codes: torch.Tensor, shape=(B, 1, T), dtype=torch.int, semantic codes
        - acoustic_codes: torch.Tensor, shape=(B, num_vq-1, T), dtype=torch.int, acoustic codes
        """
        audio = audio.to(self.device)
        if self.semantic_model_type == "sensevoice":
            feat = self._extract_sensevoice_semantic_repr(audio)
        else:
            audio_16k = torchaudio.functional.resample(audio, 24000, 16000)

            feature_extractor = self.semantic_cfg.feature_extractor

            if audio.shape[0] > 1:
                input_features_list = []
                attention_mask_list = []
                for i in range(audio.shape[0]):
                    inputs = feature_extractor(
                        audio_16k[i].cpu(), sampling_rate=16000, return_tensors="pt"
                    )
                    input_features_list.append(inputs["input_features"][0])
                    attention_mask_list.append(inputs["attention_mask"][0])
                input_features = torch.stack(input_features_list, dim=0)
                attention_mask = torch.stack(attention_mask_list, dim=0)
            else:
                inputs = feature_extractor(
                    audio_16k.cpu(), sampling_rate=16000, return_tensors="pt"
                )
                input_features = inputs["input_features"][0]
                attention_mask = inputs["attention_mask"][0]
                input_features = input_features.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            input_features = input_features.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # by default, we use autocast for semantic feature extraction
            with self._autocast_ctx():
                feat = self._extract_semantic_code(
                    input_features, attention_mask
                ).transpose(1, 2)

                feat = torch.nn.functional.avg_pool1d(
                    feat,
                    self.model.semantic_downsample_factor,
                    self.model.semantic_downsample_factor,
                )

        with self._autocast_ctx():
            semantic_codes, acoustic_codes = self.model.encode(
                audio, num_quantizers=n_quantizers, semantic_repr=feat
            )

        return semantic_codes, acoustic_codes

    def decode_from_codes(
        self,
        semantic_codes,
        acoustic_codes,
    ):
        """
        Args:
        - semantic_codes: torch.Tensor, shape=(B, 1, T), dtype=torch.int, semantic codes
        - acoustic_codes: torch.Tensor, shape=(B, num_vq-1, T), dtype=torch.int, acoustic codes
        Returns:
        - audio: torch.Tensor, shape=(B, 1, T), dtype=torch.float32, output audio waveform
        """
        audio = self.model.decode_from_codes(semantic_codes, acoustic_codes).to(
            torch.float32
        )
        return audio

    @torch.no_grad()
    def decode(self, semantic_codes, acoustic_codes):
        """
        Args:
        - semantic_codes: torch.Tensor, shape=(B, 1, T), dtype=torch.int, semantic codes
        - acoustic_codes: torch.Tensor, shape=(B, num_vq-1, T), dtype=torch.int, acoustic codes
        Returns:
        - audio: torch.Tensor, shape=(B, 1, T), dtype=torch.float32, output audio waveform
        """
        audio = self.model.decode_from_codes(semantic_codes, acoustic_codes).to(
            torch.float32
        )
        return audio

    @torch.no_grad()
    def _extract_semantic_code(self, input_features, attention_mask):
        """
        Extract semantic code from the input features.
        Args:
        - input_features: torch.Tensor, shape=(B, T, C), dtype=torch.float32, input features
        - attention_mask: torch.Tensor, shape=(B, T), dtype=torch.int, attention mask
        Returns:
        - feat: torch.Tensor, shape=(B, T, C), dtype=torch.float32, extracted semantic code
        """
        vq_emb = self.semantic_cfg["semantic_model"](
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[self.semantic_cfg["output_idx"]]  # (B, T, C)

        if (
            hasattr(self.semantic_cfg, "skip_semantic_normalize")
            and self.semantic_cfg.skip_semantic_normalize
        ):  # skip normalization
            pass
        else:
            feat = (feat - self.semantic_cfg["mean"]) / self.semantic_cfg["std"]
        return feat


@torch.no_grad()
def infer(audio, model=None, num_quantizers=8):
    audio = audio.reshape(1, 1, -1).cpu()
    out, codes = model.inference(audio, n_quantizers=num_quantizers)
    out = pad_to_length(out, audio.shape[-1])
    return out, codes


def pad_to_length(x, length, pad_value=0):
    # Get the current size along the last dimension
    current_length = x.shape[-1]

    # If the length is greater than current_length, we need to pad
    if length > current_length:
        pad_amount = length - current_length
        # Pad on the last dimension (right side), keeping all other dimensions the same
        x_padded = F.pad(x, (0, pad_amount), value=pad_value)
    else:
        # If no padding is required, simply slice the tensor
        x_padded = x[..., :length]

    return x_padded
