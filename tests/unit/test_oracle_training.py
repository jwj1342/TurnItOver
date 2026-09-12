import pytest


def test_weighted_loss_excludes_prompt_and_matches_empirical_distribution():
    torch = pytest.importorskip('torch')
    from turnitover.oracle.training import weighted_assistant_loss
    logits = torch.tensor([[[9.,-9.],[2.,0.],[0.,2.]], [[-9.,9.],[2.,0.],[0.,2.]]], requires_grad=True)
    labels = torch.tensor([[-100,-100,0],[-100,-100,1]])
    loss = weighted_assistant_loss(logits, labels, [2/3,1/3])
    expected = -(2/3*torch.log_softmax(torch.tensor([2.,0.]),0)[0]+1/3*torch.log_softmax(torch.tensor([2.,0.]),0)[1])
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.all(logits.grad[:,0]==0)
    with pytest.raises(ValueError):
        weighted_assistant_loss(logits,labels,[0.,1.])
