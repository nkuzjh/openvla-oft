"""Native OpenVLA-OFT model adapter for CSGO Benchmark v2 localization.

The adapter deliberately keeps the model path close to ``vla-scripts/finetune.py``:
the pretrained Prismatic model is loaded with the repository's OFT-aware HF
class, all-linear LoRA is applied with PEFT, and the native
``L1RegressionActionHead`` consumes the action-token hidden states.  CSGO has a
single five-value action, so the action-token suffix is always five fixed
placeholders followed by the stop token.  Ground-truth pose values are carried
in a separate ``actions`` tensor and are never used to construct input IDs or
labels.

Imports in this module are intentionally free of the RLDS/TF data package.  A
CSGO process sets ``OPENVLA_ROBOT_PLATFORM=CSGO`` before importing this module,
which makes the native constants explicit without changing robot defaults.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# The native action head and Prismatic HF model read constants at import time.
# Set the opt-in default here as a second line of defence for callers that
# import ``csgo_seen10.model`` directly instead of through train_seen10.py.
os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "CSGO")

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch import nn
from transformers import AutoConfig, AutoModelForVision2Seq, AutoTokenizer

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
)

if (NUM_ACTIONS_CHUNK, ACTION_DIM) != (1, 5):
    raise RuntimeError(
        "CSGO Seen-10 requires native constants NUM_ACTIONS_CHUNK=1 and ACTION_DIM=5; "
        f"got {NUM_ACTIONS_CHUNK=} {ACTION_DIM=} (set OPENVLA_ROBOT_PLATFORM=CSGO before import)"
    )


CSGO_ACTION_DIM = 5
CSGO_HORIZON = 1
# The action head mask only needs an action-vocabulary token.  This value is
# independent of every target pose and is valid for the original Llama vocab.
DUMMY_ACTION_TOKEN_ID = ACTION_TOKEN_BEGIN_IDX + 1
EMPTY_TOKEN_ID = 29871


def unwrap_module(module: nn.Module) -> nn.Module:
    """Return a DDP/PEFT wrapped module's underlying module when present."""

    return module.module if hasattr(module, "module") else module


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _resolve_model_path(model_path: str | os.PathLike[str]) -> str:
    """Resolve a local snapshot or download the named HF model snapshot."""

    requested = str(model_path).rstrip("/")
    local = Path(requested).expanduser()
    if local.is_dir():
        return str(local.resolve())
    # Keep the native ``openvla/openvla-7b`` contract for callers that do not
    # provide the local mirror.  Downloading is performed only when needed.
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=requested)


def _load_local_processor(model_path: str) -> PrismaticProcessor:
    """Load the native Prismatic processor without importing RLDS modules."""

    image_processor = PrismaticImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)


def _load_local_vla(model_path: str, *, torch_dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Load the actual OpenVLA-7B weights with the repository's OFT model class.

    The public snapshot contains an older ``modeling_prismatic.py`` for remote
    code loading.  Registering the local class here is intentional: the local
    implementation contains the native multi-image channel-stack path used by
    OFT and avoids mutating the downloaded checkpoint's ``config.json``.
    """

    # Registering is harmless when another caller registered the same classes;
    # the explicit class below is what controls construction in this process.
    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    except ValueError:
        # Transformers raises when the same key is already registered.  The
        # loaded config/class are still usable, so only ignore that duplicate.
        pass

    config = OpenVLAConfig.from_pretrained(model_path)
    # The OFT fork selects its bidirectional path through SDPA.  Set the
    # implementation on both the wrapper and nested text config before the
    # language model is constructed.  This is deliberately an explicit SDPA
    # request; silently accepting the stock eager/causal implementation would
    # change the semantics of parallel action decoding.
    for candidate in (config, getattr(config, "text_config", None)):
        if candidate is not None:
            try:
                candidate._attn_implementation = "sdpa"
            except AttributeError:
                # Very old fork revisions expose the field as a plain private
                # attribute only after construction; the post-load assertion
                # below still verifies the actual hook.
                candidate._attn_implementation = "sdpa"
    vla = OpenVLAForActionPrediction.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    return vla


def _load_action_head(
    vla: nn.Module,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    *,
    device: torch.device | str = "cpu",
) -> L1RegressionActionHead:
    base = unwrap_module(vla)
    head = L1RegressionActionHead(
        input_dim=int(base.llm_dim),
        hidden_dim=int(base.llm_dim),
        action_dim=ACTION_DIM,
    ).to(device=device, dtype=torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32)
    if checkpoint_dir is not None:
        path = _find_component_checkpoint(Path(checkpoint_dir), "action_head")
        if path is None:
            raise FileNotFoundError(f"Checkpoint {checkpoint_dir} has no action_head state")
        state = torch.load(path, map_location="cpu", weights_only=True)
        head.load_state_dict(_strip_ddp_prefix(state))
    return head


def _strip_ddp_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def _find_component_checkpoint(checkpoint_dir: Path, component: str) -> Path | None:
    """Find native-style component state names in a checkpoint directory."""

    if not checkpoint_dir.is_dir():
        return None
    candidates = sorted(checkpoint_dir.glob(f"{component}--*_checkpoint.pt"))
    candidates += [checkpoint_dir / f"{component}.pt", checkpoint_dir / f"{component}.pth"]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _find_adapter_dir(checkpoint_dir: Path) -> Path | None:
    for candidate in (checkpoint_dir / "lora_adapter", checkpoint_dir):
        if (candidate / "adapter_config.json").is_file():
            return candidate
    return None


@dataclass
class ModelBundle:
    """Model components and metadata shared by training and inference."""

    vla: nn.Module
    action_head: nn.Module
    processor: PrismaticProcessor
    model_path: str
    checkpoint_dir: str | None = None
    base_vla: nn.Module | None = None

    @property
    def llm_dim(self) -> int:
        return int(unwrap_module(self.vla).llm_dim)

    @property
    def num_patches(self) -> int:
        backbone = unwrap_module(self.vla).vision_backbone
        return int(backbone.get_num_patches() * backbone.get_num_images_in_input())


def _enable_native_gradient_checkpointing(vla: nn.Module) -> None:
    """Enable checkpointing on the language model itself, as native OFT does."""

    base = unwrap_module(vla)
    language_model = getattr(base, "language_model", None)
    if language_model is None or not hasattr(language_model, "gradient_checkpointing_enable"):
        raise RuntimeError("Loaded OpenVLA language_model has no gradient_checkpointing_enable()")
    # The native OFT training path uses non-reentrant checkpointing.  The
    # reentrant default can mark a frozen/LoRA branch ready twice under DDP.
    language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # PEFT's checkpointed blocks need input embeddings to retain a gradient
    # path to LoRA layers.  This is the standard HF helper, not a VLA wrapper.
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    if hasattr(language_model, "config"):
        language_model.config.use_cache = False


def _assert_oft_bidirectional(vla: nn.Module) -> None:
    """Fail early when the loaded LLM is not the OFT bidirectional fork.

    The hook lives on ``language_model.model`` for LlamaForCausalLM in the
    original OFT fork (some revisions also expose it on the outer object).
    Checking only the outer object lets stock Transformers through because it
    has no hook there, so require either the fork marker or source that
    explicitly returns an unmasked SDPA path.
    """

    base = unwrap_module(vla)
    language_model = getattr(base, "language_model", None)
    if language_model is None:
        raise RuntimeError("OpenVLA model has no language_model")
    config = getattr(language_model, "config", None)
    implementation = getattr(config, "_attn_implementation", None)
    if implementation != "sdpa":
        raise RuntimeError(
            "CSGO Seen-10 requires the original OpenVLA-OFT SDPA implementation; "
            f"loaded attention implementation={implementation!r}"
        )
    # OFT's custom Llama implementation may expose this marker.
    sources: list[str] = []
    try:
        import inspect

        for owner in (getattr(language_model, "model", None), language_model):
            update_mask = getattr(owner, "_update_causal_mask", None)
            if update_mask is None:
                continue
            try:
                sources.append(inspect.getsource(update_mask))
            except (OSError, TypeError):
                continue
    except ImportError:
        sources = []
    if not sources:
        raise RuntimeError(
            "Loaded SDPA language model has no inspectable _update_causal_mask hook; "
            "refusing to run without proof of the OpenVLA-OFT bidirectional fork"
        )
    source = re.sub(r"\s+", "", "\n".join(sources))
    if "returnNone" not in source and "is_causal=False" not in source:
        raise RuntimeError(
            "Loaded transformers Llama implementation does not expose the OFT bidirectional "
            "causal-mask behavior (expected return None or is_causal=False)"
        )

    # A stock SDPA implementation also has a conditional ``return None`` for
    # padding-free causal decoding.  Exercise the loaded fork itself through a
    # tiny CPU language model so a source substring cannot accidentally certify
    # stock causal attention.  In OFT, changing the next valid token changes the
    # first token state, while changing a masked padding token does not.
    try:
        import copy

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            tiny_config = copy.deepcopy(config)
            tiny_config.vocab_size = 128
            tiny_config.hidden_size = 64
            tiny_config.intermediate_size = 128
            tiny_config.num_hidden_layers = 1
            tiny_config.num_attention_heads = 4
            if hasattr(tiny_config, "num_key_value_heads"):
                tiny_config.num_key_value_heads = 4
            tiny_config.max_position_embeddings = 32
            tiny_config.pad_token_id = 0
            tiny_config._attn_implementation = "sdpa"
            tiny_model = type(language_model)(tiny_config).to("cpu").eval()

            def _hidden(ids: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
                result = tiny_model(
                    input_ids=ids,
                    attention_mask=attention,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                return result.hidden_states[-1]

            full_mask = torch.ones((1, 2), dtype=torch.long)
            first_a = _hidden(torch.tensor([[1, 5]]), full_mask)[:, 0]
            first_b = _hidden(torch.tensor([[1, 6]]), full_mask)[:, 0]
            valid_delta = float((first_a - first_b).abs().max().item())

            padded_mask = torch.tensor([[1, 0]], dtype=torch.long)
            padded_a = _hidden(torch.tensor([[1, 5]]), padded_mask)[:, 0]
            padded_b = _hidden(torch.tensor([[1, 6]]), padded_mask)[:, 0]
            padding_delta = float((padded_a - padded_b).abs().max().item())
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise RuntimeError(
            "Unable to verify the original OpenVLA-OFT bidirectional SDPA behavior with a tiny model"
        ) from exc
    if valid_delta <= 1e-6 or padding_delta > 1e-5:
        raise RuntimeError(
            "Loaded transformers attention is not the OFT bidirectional SDPA behavior: "
            f"next_valid_delta={valid_delta:.3g}, masked_padding_delta={padding_delta:.3g}"
        )


def create_model(
    model_path: str | os.PathLike[str],
    *,
    device: torch.device | str,
    use_lora: bool = True,
    lora_rank: int = 32,
    lora_dropout: float = 0.0,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    gradient_checkpointing: bool = True,
) -> ModelBundle:
    """Create the pretrained model, CSGO two-image input path and action head."""

    resolved_model_path = _resolve_model_path(model_path)
    processor = _load_local_processor(resolved_model_path)
    base_vla = _load_local_vla(resolved_model_path)
    base_vla.vision_backbone.set_num_images_in_input(2)
    _assert_oft_bidirectional(base_vla)

    vla: nn.Module = base_vla
    adapter_dir = _find_adapter_dir(Path(checkpoint_dir)) if checkpoint_dir is not None else None
    if adapter_dir is not None:
        vla = PeftModel.from_pretrained(vla, str(adapter_dir), is_trainable=True)
    elif use_lora:
        lora_config = LoraConfig(
            r=int(lora_rank),
            lora_alpha=min(int(lora_rank), 16),
            lora_dropout=float(lora_dropout),
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)

    if gradient_checkpointing:
        _enable_native_gradient_checkpointing(vla)

    device = torch.device(device)
    vla = vla.to(device)
    action_head = _load_action_head(vla, checkpoint_dir, device=device)
    return ModelBundle(
        vla=vla,
        action_head=action_head,
        processor=processor,
        model_path=resolved_model_path,
        checkpoint_dir=str(Path(checkpoint_dir).resolve()) if checkpoint_dir is not None else None,
        base_vla=base_vla,
    )


def _prompt_ids(tokenizer: Any, instruction: str) -> list[int]:
    """Build the native PurePromptBuilder prompt up to ``Out:``."""

    builder = PurePromptBuilder("openvla")
    builder.add_turn("human", f"What action should the robot take to {instruction.lower()}?")
    encoded = tokenizer(builder.get_prompt(), add_special_tokens=True).input_ids
    # Llama tokenization of a trailing ``Out: `` normally emits this empty
    # token.  Match native predict_action/finetune behavior explicitly.
    if not encoded or encoded[-1] != EMPTY_TOKEN_ID:
        encoded.append(EMPTY_TOKEN_ID)
    return [int(value) for value in encoded]


def _sample_instance(
    sample: Mapping[str, Any], processor: PrismaticProcessor, *, include_target: bool
) -> dict[str, Any]:
    """Load one dataset metadata record into the native model-facing format."""

    fpv_path = sample.get("fpv_path", sample.get("image_path"))
    radar_path = sample.get("map_path", sample.get("radar_path"))
    if fpv_path is None or radar_path is None:
        raise KeyError("Seen-10 sample needs fpv_path/image_path and map_path/radar_path")
    with Image.open(fpv_path) as fpv_image:
        fpv = processor.image_processor.apply_transform(fpv_image.convert("RGB"))
    with Image.open(radar_path) as radar_image:
        radar = processor.image_processor.apply_transform(radar_image.convert("RGB"))
    if fpv.ndim != 3 or radar.ndim != 3:
        raise ValueError(f"Expected image tensors [C,H,W], got {tuple(fpv.shape)} and {tuple(radar.shape)}")
    if fpv.shape[0] != 6 or radar.shape[0] != 6:
        raise ValueError(
            "CSGO Seen-10 expects each fused Prismatic image to have six channels "
            f"(SigLIP+DINO); got {fpv.shape[0]} and {radar.shape[0]}"
        )
    # Each fused Prismatic image has SigLIP+DINO channels (6); concatenating
    # two images therefore gives the required native 12-channel input.
    pixel_values = torch.cat((fpv, radar), dim=0).contiguous()

    tokenizer = processor.tokenizer
    base_ids = _prompt_ids(tokenizer, str(sample["instruction"]))
    input_ids = torch.tensor(base_ids + [1] * CSGO_ACTION_DIM + [STOP_INDEX], dtype=torch.long)
    labels = torch.tensor(
        [IGNORE_INDEX] * len(base_ids) + [DUMMY_ACTION_TOKEN_ID] * CSGO_ACTION_DIM + [STOP_INDEX],
        dtype=torch.long,
    )
    item = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels,
        "sample_id": str(sample["sample_id"]),
        "map_name": str(sample["map_name"]),
        "file_frame": str(sample.get("file_frame", str(sample["sample_id"]).split("/")[-1])),
        "instruction": str(sample["instruction"]),
        "clip_id": sample.get("clip_id"),
        "frame_index": sample.get("frame_index"),
    }
    if include_target:
        target = sample.get("target_pose", sample.get("target"))
        if target is None:
            raise KeyError("Training sample has no target_pose/target")
        if len(target) != CSGO_ACTION_DIM:
            raise ValueError(f"Expected five normalized target values, got {len(target)}")
        # Keep action labels separate from token labels.  The latter remain
        # fixed dummy tokens even for training records.
        item["actions"] = torch.tensor([[float(value) for value in target]], dtype=torch.float32)
    return item


def collate_samples(
    samples: Sequence[Mapping[str, Any]],
    processor: PrismaticProcessor,
    *,
    include_targets: bool,
) -> dict[str, Any]:
    """Collate manifest records into a native OpenVLA batch."""

    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    instances = [_sample_instance(sample, processor, include_target=include_targets) for sample in samples]
    pad_id = int(processor.tokenizer.pad_token_id)
    if processor.tokenizer.padding_side != "right":
        raise ValueError("CSGO Seen-10 native collator requires right padding")
    max_len = max(item["input_ids"].numel() for item in instances)
    input_ids, labels, pixel_values = [], [], []
    for item in instances:
        pad_len = max_len - item["input_ids"].numel()
        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_id))
        labels.append(F.pad(item["labels"], (0, pad_len), value=IGNORE_INDEX))
        pixel_values.append(item["pixel_values"])
    output: dict[str, Any] = {
        "pixel_values": torch.stack(pixel_values, dim=0),
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(input_ids, dim=0).ne(pad_id),
        "labels": torch.stack(labels, dim=0),
        "sample_ids": [item["sample_id"] for item in instances],
        "map_names": [item["map_name"] for item in instances],
        "file_frames": [item["file_frame"] for item in instances],
        "clip_ids": [item.get("clip_id") for item in instances],
        "frame_indices": [item.get("frame_index") for item in instances],
    }
    if include_targets:
        # Each item is [horizon=1, action_dim=5].  ``cat(..., dim=0)`` would
        # collapse the horizon and make the L1 target [B, 5], which silently
        # broadcasts against predictions [B, 1, 5] for batch size > 1.
        output["actions"] = torch.stack([item["actions"] for item in instances], dim=0)
    return output


def _mask_and_action_hidden_states(
    output: Any,
    labels: torch.Tensor,
    *,
    num_patches: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract action hidden states using the native one-token shift convention."""

    # Native run_forward_pass indexes labels[:, 1:] against hidden states
    # [num_patches:-1].  Keep both tensors on the model device before boolean
    # indexing; CPU masks against CUDA hidden states are invalid.
    token_labels = labels[:, 1:]
    current_mask = get_current_action_mask(token_labels)
    next_mask = get_next_actions_mask(token_labels)
    hidden = output.hidden_states[-1]
    text_hidden = hidden[:, num_patches:-1]
    all_mask = current_mask | next_mask
    batch_size = labels.shape[0]
    expected = batch_size * NUM_ACTIONS_CHUNK * ACTION_DIM
    if int(all_mask.sum().item()) != expected:
        raise ValueError(
            "Native action mask does not contain exactly the CSGO action suffix: "
            f"expected {expected}, got {int(all_mask.sum().item())}"
        )
    action_hidden = text_hidden[all_mask].reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
    return action_hidden.to(torch.bfloat16), current_mask, next_mask


def forward_action(
    bundle: ModelBundle,
    batch: Mapping[str, Any],
    *,
    device: torch.device | str,
    train: bool,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Run the native multimodal forward and return prediction/action loss."""

    device = torch.device(device)
    labels = batch["labels"].to(device=device)
    inputs = batch["input_ids"].to(device=device)
    attention_mask = batch["attention_mask"].to(device=device)
    pixels = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
    with _autocast(device):
        output = bundle.vla(
            input_ids=inputs,
            attention_mask=attention_mask,
            pixel_values=pixels,
            labels=labels,
            output_hidden_states=True,
            proprio=None,
            proprio_projector=None,
            use_film=False,
        )
        action_hidden, current_mask, next_mask = _mask_and_action_hidden_states(
            output, labels, num_patches=bundle.num_patches
        )
        # Call the head wrapper itself so DDP enters its reducer and synchronizes
        # head gradients.  The native head exposes ``forward`` as the same
        # operation as ``predict_action``.
        head_parameters = list(unwrap_module(bundle.action_head).parameters())
        head_dtype = head_parameters[0].dtype if head_parameters else action_hidden.dtype
        normalized_actions = bundle.action_head(action_hidden.to(dtype=head_dtype))
        normalized_actions = normalized_actions.reshape(-1, CSGO_HORIZON, CSGO_ACTION_DIM)

        if "actions" in batch:
            targets = batch["actions"].to(device=device, dtype=normalized_actions.dtype)
            loss = F.l1_loss(targets, normalized_actions)
            current_loss = F.l1_loss(targets[:, 0], normalized_actions[:, 0])
            # Horizon one has no future action.  ``F.l1_loss`` on two empty
            # tensors returns NaN, so expose a finite zero metric explicitly.
            next_loss = torch.zeros((), device=device, dtype=loss.dtype)
            metrics = {
                "loss_value": float(loss.detach().float().item()),
                "curr_action_l1_loss": float(current_loss.detach().float().item()),
                "next_actions_l1_loss": float(next_loss.item()),
                "action_count": float(targets.numel()),
                "current_action_tokens": float(current_mask.sum().item()),
                "next_action_tokens": float(next_mask.sum().item()),
            }
        else:
            loss = torch.zeros((), device=device, dtype=normalized_actions.dtype)
            metrics = {
                "loss_value": 0.0,
                "curr_action_l1_loss": 0.0,
                "next_actions_l1_loss": 0.0,
                "action_count": 0.0,
                "current_action_tokens": float(current_mask.sum().item()),
                "next_action_tokens": float(next_mask.sum().item()),
            }
    return loss, metrics, normalized_actions.detach()


def native_run_forward_pass(
    bundle: ModelBundle,
    batch: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Compatibility name for the native finetune forward path."""

    return forward_action(bundle, batch, device=device, train=True)


def predict_normalized(
    bundle: ModelBundle,
    batch: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Predict normalized [x,y,z,pitch,yaw] without requiring target fields."""

    was_training_vla = bundle.vla.training
    was_training_head = bundle.action_head.training
    bundle.vla.eval()
    bundle.action_head.eval()
    with torch.no_grad():
        _, _, predictions = forward_action(bundle, batch, device=device, train=False)
    if was_training_vla:
        bundle.vla.train()
    if was_training_head:
        bundle.action_head.train()
    return predictions[:, 0].float().cpu()


def save_component_checkpoint(bundle: ModelBundle, checkpoint_dir: Path, step: int) -> None:
    """Save PEFT adapter, processor and native action-head state."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    vla = unwrap_module(bundle.vla)
    if isinstance(vla, PeftModel):
        vla.save_pretrained(adapter_dir)
    else:
        raise RuntimeError("CSGO Seen-10 checkpoints require the native PEFT LoRA adapter")
    bundle.processor.save_pretrained(checkpoint_dir)
    torch.save(bundle.action_head.state_dict(), checkpoint_dir / f"action_head--{step}_checkpoint.pt")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_provenance(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return lightweight, deterministic provenance for a pretrained snapshot."""

    path = Path(model_path).expanduser().resolve()
    config = path / "config.json"
    index = path / "model.safetensors.index.json"
    weight_files = sorted(path.glob("model*.safetensors"))
    return {
        "model_name": "openvla/openvla-7b",
        "model_path": str(path),
        "config_sha256": sha256_file(config) if config.is_file() else None,
        # A sharded snapshot must not be reported as if one shard were the
        # complete model.  The full weights are intentionally left unhashed;
        # the index digest and listed shard names still make provenance useful.
        "weights_sha256": None,
        "weights_sha256_file": None,
        "weights_index_sha256": sha256_file(index) if index.is_file() else None,
        "weight_files": [item.name for item in weight_files],
        "hf_revision": os.environ.get("OPENVLA_REVISION"),
    }


__all__ = [
    "CSGO_ACTION_DIM",
    "CSGO_HORIZON",
    "DUMMY_ACTION_TOKEN_ID",
    "ModelBundle",
    "collate_samples",
    "create_model",
    "forward_action",
    "model_provenance",
    "native_run_forward_pass",
    "predict_normalized",
    "save_component_checkpoint",
    "unwrap_module",
]
