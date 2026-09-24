"""Small CPU checks for aligned model wiring; no pretrained weights are loaded."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from csgo_seen10.action_normalization import ActionNormalization, fit_seen_train_stats
from csgo_seen10 import model as adapter


class TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.images = 1

    def set_num_images_in_input(self, count: int) -> None:
        self.images = count

    def get_num_patches(self) -> int:
        return 0

    def get_num_images_in_input(self) -> int:
        return self.images


class TinyVLA(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self) -> None:
        super().__init__(PretrainedConfig())
        self.llm_dim = 4
        self.vision_backbone = TinyVision()
        self.projector = nn.Linear(4, 4)
        self.language_model = nn.Module()
        self.language_model.embed = nn.Embedding(32064, 4)
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([
            nn.ModuleDict({"q_proj": nn.Linear(4, 4), "down_proj": nn.Linear(4, 4)})
        ])
        self.language_model.lm_head = nn.Linear(4, 32064, bias=False)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.lm_head

    def forward(self, *, input_ids: torch.Tensor, **kwargs):
        hidden = self.language_model.embed(input_ids)
        hidden = self.language_model.model.layers[0]["q_proj"](hidden)
        hidden = self.language_model.model.layers[0]["down_proj"](hidden)
        return SimpleNamespace(hidden_states=(hidden,))


def _normalization() -> ActionNormalization:
    records = [
        {"sample_id": f"s{i}", "split": "seen_train", "target_pose": [i / 4] * 5}
        for i in range(4)
    ]
    stats = fit_seen_train_stats(records, "0" * 64, expected_count=4)
    return ActionNormalization("bounds_q99", stats)


def test_aligned_lora_freezes_vision_projector_and_keeps_lm_head():
    norm = _normalization()
    with tempfile.TemporaryDirectory() as root, \
        patch.object(adapter, "_load_local_processor", return_value=SimpleNamespace()), \
        patch.object(adapter, "_load_local_vla", return_value=TinyVLA()), \
        patch.object(adapter, "_assert_oft_bidirectional"):
        bundle = adapter.create_model(
            root, device="cpu", lora_rank=2, lora_alpha=1,
            gradient_checkpointing=False, freeze_vision=True, freeze_vl_projector=True,
            lora_scope="official_all_linear_excluding_frozen_modules",
            action_normalization=norm, recipe_id="aligned_test",
        )
    adapter_names = [name for name, _ in bundle.vla.named_parameters() if "lora_A" in name]
    assert len(adapter_names) == 3
    assert any("language_model.lm_head" in name for name in adapter_names)
    assert all("vision_backbone" not in name and "projector" not in name for name in adapter_names)
    assert not any(p.requires_grad for p in bundle.base_vla.vision_backbone.parameters())
    assert not any(p.requires_grad for p in bundle.base_vla.projector.parameters())

    labels = torch.tensor([[adapter.IGNORE_INDEX] * 3 + [adapter.DUMMY_ACTION_TOKEN_ID] * 5 + [adapter.STOP_INDEX]])
    batch = {
        "labels": labels,
        "input_ids": torch.tensor([[1, 2, 3, 1, 1, 1, 1, 1, 1]]),
        "attention_mask": torch.ones((1, 9), dtype=torch.bool),
        "pixel_values": torch.zeros((1, 12, 2, 2)),
        "actions": torch.tensor([[[0.5] * 5]]),
    }
    bundle.vla.train()
    loss, metrics, prediction = adapter.forward_action(bundle, batch, device="cpu", train=True)
    assert not bundle.base_vla.vision_backbone.training
    assert not bundle.base_vla.projector.training
    assert prediction.shape == (1, 1, 5)
    assert torch.isfinite(loss)
    assert metrics["loss_value"] >= 0 and metrics["external_loss_value"] >= 0
    loss.backward()
    assert bundle.action_head.model.fc1.weight.grad is not None
    assert any("q_proj" in name and p.grad is not None for name, p in bundle.vla.named_parameters())
    assert all(p.grad is None for name, p in bundle.vla.named_parameters() if "lm_head.lora_" in name)
    with torch.no_grad():
        bundle.action_head.model.fc2.weight.zero_()
        bundle.action_head.model.fc2.bias.fill_(10.0)
        _, _, unbounded = adapter.forward_action(bundle, batch, device="cpu", train=False)
    assert bool(torch.all(unbounded > torch.tensor(norm.stats["q99"])))


def test_aligned_recipe_rejects_missing_or_mismatched_stats():
    norm = _normalization()
    expected = {
        "recipe_id": "aligned_test", "lora_scope": "official_all_linear_excluding_frozen_modules",
        "freeze_vision": True, "freeze_vl_projector": True,
        "normalization_sha256": adapter._normalization_digest(norm),
    }
    with tempfile.TemporaryDirectory() as root:
        tmp_path = Path(root)
        with unittest.TestCase().assertRaises(FileNotFoundError):
            adapter._validate_checkpoint_recipe(tmp_path, expected, norm)
        (tmp_path / "model_recipe.json").write_text(json.dumps(expected), encoding="utf-8")
        norm.save(tmp_path / "action_normalization.json")
        adapter._validate_checkpoint_recipe(tmp_path, expected, norm)
        tampered = dict(expected, freeze_vision=False)
        with unittest.TestCase().assertRaises(ValueError):
            adapter._validate_checkpoint_recipe(tmp_path, tampered, norm)
    assert adapter._normalization_digest(None) == adapter._normalization_digest(ActionNormalization())


class AlignedModelTests(unittest.TestCase):
    def test_model_wiring(self):
        test_aligned_lora_freezes_vision_projector_and_keeps_lm_head()

    def test_checkpoint_recipe(self):
        test_aligned_recipe_rejects_missing_or_mismatched_stats()

    def test_collator_augments_only_training_images(self):
        class Tokenizer:
            pad_token_id = 0
            padding_side = "right"

            def __call__(self, *_args, **_kwargs):
                return SimpleNamespace(input_ids=[1, 2, 3])

        class ImageProcessor:
            def apply_transform(self, image):
                values = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
                return torch.cat((values, values), dim=0)

        processor = SimpleNamespace(tokenizer=Tokenizer(), image_processor=ImageProcessor())
        with tempfile.TemporaryDirectory() as root:
            fpv = Path(root) / "fpv.png"
            radar = Path(root) / "radar.png"
            Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(fpv)
            Image.fromarray(np.full((16, 16, 3), 97, dtype=np.uint8)).save(radar)
            sample = {
                "fpv_path": str(fpv), "map_path": str(radar), "sample_id": "map/frame",
                "map_name": "map", "instruction": "locate", "target_pose": [0.1] * 5,
                "_augmentation_epoch": 3,
            }
            train = adapter.collate_samples(
                [sample], processor, include_targets=True, training=True,
                fpv_augmentation="oft_photometric_only",
                radar_augmentation="oft_photometric_only", augmentation_seed=4,
            )
            validation = adapter.collate_samples(
                [sample], processor, include_targets=True, training=False,
                fpv_augmentation="oft_photometric_only",
                radar_augmentation="oft_photometric_only", augmentation_seed=4,
            )
        self.assertEqual(tuple(train["pixel_values"].shape), (1, 12, 16, 16))
        self.assertTrue(torch.equal(train["actions"], validation["actions"]))
        self.assertFalse(torch.equal(train["pixel_values"], validation["pixel_values"]))
        self.assertTrue(torch.equal(train["pixel_values"][:, :3], train["pixel_values"][:, 3:6]))
        self.assertTrue(torch.equal(train["pixel_values"][:, 6:9], train["pixel_values"][:, 9:12]))
