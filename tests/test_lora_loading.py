import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from wan_va.modules.lora import merge_lora_adapter_into_model


def _write_adapter(
    root,
    state,
    matched_modules,
    *,
    rank=2,
    alpha=4.0,
):
    adapter_dir = root / "lora_adapter"
    adapter_dir.mkdir()
    save_file(state, adapter_dir / "adapter_model.safetensors")
    with open(adapter_dir / "adapter_config.json", "w") as stream:
        json.dump(
            {
                "format": "lingbot-va-lora",
                "base_model_name_or_path": "/models/lingbot-va-base",
                "target_modules": matched_modules,
                "rank": rank,
                "alpha": alpha,
                "matched_modules": matched_modules,
            },
            stream,
        )
    return adapter_dir


def test_merge_lora_checkpoint_root_and_reject_double_merge(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        model[0].weight.zero_()
    a_weight = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    b_weight = torch.tensor([[2.0, -1.0], [0.25, 3.0]])
    _write_adapter(
        tmp_path,
        {
            "0.lora_A.weight": a_weight,
            "0.lora_B.weight": b_weight,
        },
        ["0"],
    )

    info = merge_lora_adapter_into_model(model, tmp_path)

    torch.testing.assert_close(model[0].weight, (b_weight @ a_weight) * 2.0)
    assert info["scale"] == 2.0
    assert info["matched_modules"] == ["0"]
    with pytest.raises(RuntimeError, match="already been merged"):
        merge_lora_adapter_into_model(model, tmp_path)


class _TwoLinears(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(3, 2, bias=False)
        self.second = nn.Linear(4, 2, bias=False)


def test_invalid_late_tensor_does_not_partially_mutate_model(tmp_path):
    model = _TwoLinears()
    before_first = model.first.weight.detach().clone()
    before_second = model.second.weight.detach().clone()
    _write_adapter(
        tmp_path,
        {
            "first.lora_A.weight": torch.ones(2, 3),
            "first.lora_B.weight": torch.ones(2, 2),
            "second.lora_A.weight": torch.ones(2, 3),
            "second.lora_B.weight": torch.ones(2, 2),
        },
        ["first", "second"],
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        merge_lora_adapter_into_model(model, tmp_path)

    torch.testing.assert_close(model.first.weight, before_first)
    torch.testing.assert_close(model.second.weight, before_second)


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("inf")])
def test_rejects_invalid_alpha(tmp_path, alpha):
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    _write_adapter(
        tmp_path,
        {
            "0.lora_A.weight": torch.ones(2, 3),
            "0.lora_B.weight": torch.ones(2, 2),
        },
        ["0"],
        alpha=alpha,
    )
    before = model[0].weight.detach().clone()

    with pytest.raises(ValueError, match="rank and alpha"):
        merge_lora_adapter_into_model(model, tmp_path)

    torch.testing.assert_close(model[0].weight, before)


def test_rejects_nonfinite_adapter_without_mutation(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    a_weight = torch.ones(2, 3)
    a_weight[0, 0] = float("nan")
    _write_adapter(
        tmp_path,
        {
            "0.lora_A.weight": a_weight,
            "0.lora_B.weight": torch.ones(2, 2),
        },
        ["0"],
    )
    before = model[0].weight.detach().clone()

    with pytest.raises(ValueError, match="non-finite"):
        merge_lora_adapter_into_model(model, tmp_path)

    torch.testing.assert_close(model[0].weight, before)
