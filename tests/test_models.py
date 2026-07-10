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
  - physical non-negativity constraint
  - the model factory (get_model) wires everything correctly
"""
import pytest
import torch
from models import get_model, ARCHITECTURES


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
    assert torch.all(out >= 0), "Physical constraint violated: negative precipitation"


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
    assert torch.all(out >= 0)


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
