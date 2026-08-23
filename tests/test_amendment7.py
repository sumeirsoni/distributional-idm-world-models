
import torch

from distributional_idm.models.flow import FlowIDM, MonotoneSplineFlow


def test_flow_is_monotone_and_invertible() -> None:
    torch.manual_seed(0)
    features = torch.randn(64, 8)
    flow = MonotoneSplineFlow(8)
    u = torch.randn(64)
    with torch.no_grad():
        x_edges, y_edges, slopes = flow.params(features)
        from distributional_idm.models.flow import (
            _hermite_eval,
            _hermite_invert,
        )

        x = _hermite_invert(u, y_edges, slopes, x_edges)
        y_back, dydx = _hermite_eval(x, y_edges, slopes, x_edges)
    assert torch.allclose(y_back, u, atol=1e-3)
    assert (dydx > 0).all()


def test_flow_fit_bimodal_distribution() -> None:
    torch.manual_seed(0)
    features = torch.randn(512, 4)
    actions = torch.where(torch.rand(512) < 0.5, -torch.ones(512), torch.ones(512))
    actions += 0.05 * torch.randn(512)
    model = FlowIDM(4, n_bins=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(800):
        loss = model.loss({"features": features, "actions": actions})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        outputs = model.predict(features)
        samples = model.sample(features, 16, torch.Generator().manual_seed(7))
    near_modes = ((samples.abs() - 1.0).abs() < 0.15).float().mean()
    assert near_modes > 0.85
    assert float(outputs["mean"].abs().mean()) < 0.3


def test_mdn_reparam_sample_shapes_and_grad() -> None:
    from distributional_idm.models.idm import MDNIDM

    model = MDNIDM(4, n_components=2)
    features = torch.randn(32, 4)
    samples = model.sample_reparam(features, 5, torch.Generator().manual_seed(1))
    assert samples.shape == (32, 5)


def test_cycle_loss_gradients_reach_encoder_and_predictor() -> None:
    from distributional_idm.world_model import ArmConfig, WorldModel

    for kind in ("mdn", "flow"):
        model = WorldModel(
            ArmConfig(name="c", idm_kind=kind, cycle=True, cycle_weight=1.0, idm_weight=0.3)
        )
        states = torch.randn(16)
        next_states = states + 0.25
        actions = torch.where(torch.rand(16) < 0.5, -torch.ones(16), torch.ones(16))
        z_target = torch.randn(16, 16)
        loss = model.cycle_loss(states, actions, next_states, z_target)
        assert loss is not None
        loss.backward()
        enc_grad = sum(
            float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None
        )
        pred_grad = sum(
            float(p.grad.abs().sum()) for p in model.predictor.parameters() if p.grad is not None
        )
        assert enc_grad > 0 and pred_grad > 0
