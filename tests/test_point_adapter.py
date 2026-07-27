import pytest
import torch

from nnexp.adapter import PointAdapter


def test_point_adapter_forward_shapes():
    adapter = PointAdapter(in_features=2, hidden_dim=256, d_model=2048)

    # Input format [B, 2, N] (row 0: x, row 1: y)
    batch_size = 4
    num_points = 50
    x_input = torch.randn(batch_size, 2, num_points)

    e_points = adapter(x_input)
    assert e_points.shape == (batch_size, num_points, 2048)

    # Input format [B, N, 2]
    x_transposed = torch.randn(batch_size, num_points, 2)
    e_points_trans = adapter(x_transposed)
    assert e_points_trans.shape == (batch_size, num_points, 2048)


def test_concat_embeddings():
    adapter = PointAdapter(in_features=2, hidden_dim=256, d_model=2048)

    batch_size = 2
    prompt_len = 12
    num_points = 50

    e_prompt = torch.randn(batch_size, prompt_len, 2048)
    e_points = adapter(torch.randn(batch_size, 2, num_points))

    e_total = adapter.concat_embeddings(e_prompt, e_points)
    assert e_total.shape == (batch_size, prompt_len + num_points, 2048)


def test_point_adapter_gradient_flow():
    adapter = PointAdapter(in_features=2, hidden_dim=256, d_model=2048)

    x_input = torch.randn(2, 2, 50, requires_grad=True)
    e_points = adapter(x_input)

    loss = e_points.sum()
    loss.backward()

    # Check non-null gradients on input and adapter parameters
    assert x_input.grad is not None
    for name, param in adapter.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
