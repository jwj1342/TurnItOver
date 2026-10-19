"""Check sequential variant accumulation preserves the complete-group objective."""
import pytest


def test_sequential_group_gradients_equal_packed_weighted_loss():
    torch = pytest.importorskip('torch')
    from turnitover.oracle.training import weighted_assistant_loss
    torch.manual_seed(20261017)
    parameter = torch.randn(3, 5, requires_grad=True)
    features = torch.randn(2, 4, 3)
    labels = torch.tensor([[-100, -100, 1, 2], [-100, -100, 3, -100]])
    weights = [2/3, 1/3]
    packed = weighted_assistant_loss(features @ parameter, labels, weights)
    packed.backward()
    expected = parameter.grad.clone()
    parameter.grad = None
    values = []
    for i, weight in enumerate(weights):
        loss = weighted_assistant_loss(features[i:i+1] @ parameter, labels[i:i+1], [1.])
        (loss * weight).backward()
        values.append(loss.detach()*weight)
    assert torch.allclose(sum(values), packed.detach())
    assert torch.allclose(parameter.grad, expected, atol=1e-7)
