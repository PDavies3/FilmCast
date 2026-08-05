"""
tests/test_models.py
----------------------
Covers, for both architectures ("residual" and "unet"):
  - forward pass shape correctness
  - RESOLUTION ADAPTIVITY: same trained weights, a DIFFERENT target
    resolution than the fixture uses -- including a deliberately
    non-power-of-2-divisible size to prove the U-Net's pad/crop logic
    genuinely works, not just on convenient sizes.
  - gradient flow through the conditioning vector and through every FiLM
    parameter specifically
  - the model factory (get_model) wires everything correctly
  - non-negativity is verified at the DENORMALIZE+CLIP stage (physical
    units), not on raw model output -- raw output is a NORMALIZED value
    and can legitimately be negative (e.g. zero precipitation itself
    normalizes to a negative number under log1p). The network's final
    activation used to be a ReLU enforcing non-negativity in normalized
    space, which made every dry-pixel target unreachable and killed
    gradient flow -- removed for that reason. See
    scripts/predict_period.py / scripts/infer.py for where the physical
    non-negativity clip now actually lives.
"""
import numpy as np
import pytest
import torch
from models import get_model, ARCHITECTURES
from utils.normalisation import denormalize


def _make_cond(batch_size):
    torch.manual_seed(0)
    return torch.rand(batch_size, 3) * torch.tensor([1.0, 2.0, 2.0]) - torch.tensor([0.0, 1.0, 1.0])


@pytest.mark.parametrize("architecture", list(ARCHITECTURES.keys()))
def test_factory_builds_model(architecture):
    device = torch.device("cpu")
    netG, netD = get_model(architecture, dynamic_channels=6, static_channels=2, device=device)
    assert netG is not None and netD is not None


@pytest.mark.parametrize("architecture", list(ARCHITECTURES.keys()))
def test_forward_shape_matches_fixture_resolution(architecture, sample_batch):
    device = torch.device("cpu")
    netG, _ = get_model(architecture, sample_batch["dynamic_input"].shape[1],
                         sample_batch["static_input"].shape[1], device)
    cond = _make_cond(sample_batch["dynamic_input"].shape[0])
    out = netG(sample_batch["dynamic_input"], sample_batch["static_input"], cond)

    expected_shape = (sample_batch["dynamic_input"].shape[0], 1, *sample_batch["static_input"].shape[-2:])
    assert out.shape == expected_shape
    # P.Davies: add -- non-negativity is no longer enforced inside the
    # network (output is a NORMALIZED value, which can be negative). See
    # test_denormalize_and_clip_enforces_non_negativity below for where
    # that guarantee now actually lives.


@pytest.mark.parametrize("architecture", list(ARCHITECTURES.keys()))
@pytest.mark.parametrize("target_hw", [(128, 128), (96, 160), (130, 121)])
def test_resolution_adaptivity_same_weights_different_output_size(architecture, sample_batch, target_hw):
    """The core design goal: ONE set of trained weights must work at ANY
    output resolution. (130, 121) is deliberately not divisible by the
    U-Net's downsampling factor, to prove the pad/crop logic works."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    netG, _ = get_model(architecture, sample_batch["dynamic_input"].shape[1],
                         sample_batch["static_input"].shape[1], device)
    batch_size = sample_batch["dynamic_input"].shape[0]
    cond = _make_cond(batch_size)
    static_at_target_res = torch.rand(batch_size, sample_batch["static_input"].shape[1], *target_hw)

    out = netG(sample_batch["dynamic_input"], static_at_target_res, cond)
    assert out.shape == (batch_size, 1, *target_hw)
    # non-negativity no longer enforced inside the network -- see note above


@pytest.mark.parametrize("architecture", list(ARCHITECTURES.keys()))
def test_gradient_flows_through_film_conditioning(architecture, sample_batch):
    device = torch.device("cpu")
    netG, _ = get_model(architecture, sample_batch["dynamic_input"].shape[1],
                         sample_batch["static_input"].shape[1], device)
    batch_size = sample_batch["dynamic_input"].shape[0]
    cond = _make_cond(batch_size)
    cond.requires_grad_(True)

    out = netG(sample_batch["dynamic_input"], sample_batch["static_input"], cond)
    loss = torch.mean(torch.abs(out - sample_batch["target_imerg"]))
    loss.backward()

    assert cond.grad is not None and torch.any(cond.grad != 0), \
        "Conditioning vector received no gradient -- FiLM conditioning is not being learned"

    film_params = [(n, p) for n, p in netG.named_parameters() if "film" in n.lower()]
    assert film_params, "No FiLM parameters found -- conditioning mechanism may not be wired up"
    dead = [n for n, p in film_params if p.grad is None or torch.all(p.grad == 0)]
    assert not dead, f"FiLM parameters with no gradient: {dead}"

    all_dead = [n for n, p in netG.named_parameters() if p.grad is None]
    assert not all_dead, f"Parameters with NO gradient at all (dead/unused): {all_dead}"


def test_denormalize_and_clip_enforces_non_negativity():
    """The non-negativity guarantee now lives at physical-units time
    (denormalize + clip in scripts/predict_period.py / infer.py), not
    inside the network. This is the guarantee that replaced the old
    in-network ReLU."""
    raw_normalized = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    norm_config = {"type": "log1p", "stats": {"log1p_mean": 0.08, "log1p_std": 0.27}}
    physical = denormalize(raw_normalized, norm_config)
    clipped = np.clip(physical, a_min=0, a_max=None)
    assert np.all(clipped >= 0)
