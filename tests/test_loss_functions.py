import pytest
import torch

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import NCELoss, nce_loss


@pytest.fixture
def dummy_result_batch() -> InferenceResultBatch:
    predictions = {"embedding": torch.rand(1024, 512)}
    targets = {"target": torch.zeros(1024, 512)}
    batch_dim = 1024
    result_batch = InferenceResultBatch(targets, predictions, batch_dim)
    return result_batch


# calculating asymmetric NCELoss between a batch of embeddings and itself --> zero
@pytest.mark.parametrize("key", ["embedding"])
def test_asymm_NCELoss_is_zero(dummy_result_batch, key):
    loss_func = NCELoss(prediction_key1=key, prediction_key2=key)
    assert loss_func(dummy_result_batch) <= 10e-6


# calculating nce_loss for two randomly generated batch of embeddings (manually calculated)
@pytest.mark.parametrize(
    "embedding1,embedding2",
    [
        (
            torch.Tensor([[0.38, 0.18], [0.36, 0.66], [0.72, 0.09]]),
            torch.Tensor([[0.48, 0.01], [0.54, 0.28], [0.08, 0.34]]),
        )
    ],
)
def test_nce_loss_correctness(embedding1, embedding2):
    unidirectional_loss = nce_loss(embedding1, embedding2, device="cpu", is_asymmetric=True, temperature=1.0)
    bidirectional_loss = nce_loss(embedding1, embedding2, device="cpu", is_asymmetric=False, temperature=1.0)
    assert unidirectional_loss == pytest.approx(1.1300, 0.0001)
    assert bidirectional_loss == pytest.approx(2.2577, 0.0001)


def test_clm_cross_entropy_with_ponder_loss():
    from modalities.loss_functions import CLMCrossEntropyWithPonderLoss

    target_key = "target"
    prediction_key = "prediction"
    
    loss_fn = CLMCrossEntropyWithPonderLoss(target_key=target_key, prediction_key=prediction_key)
    
    # 2 batches, 4 tokens, vocab size 8
    logits = torch.randn(2, 4, 8)
    targets = torch.randint(0, 8, (2, 4))
    ponder_loss = torch.tensor(1.23)
    
    # Pack into outputs dict
    outputs = {
        "logits": logits,
        "ponder_loss": ponder_loss,
        "metrics": {"some_metric": torch.tensor(4.56)}
    }
    
    # Pack into InferenceResultBatch
    predictions_dict = {prediction_key: outputs}
    targets_dict = {target_key: targets}
    
    batch = InferenceResultBatch(targets=targets_dict, predictions=predictions_dict, batch_dim=0)
    
    total_loss = loss_fn(batch)
    
    # Calculate expected ce loss
    ce_loss_fun = torch.nn.CrossEntropyLoss(reduction="mean")
    expected_ce = ce_loss_fun(logits.view(-1, 8), targets.view(-1))
    expected_total = expected_ce + ponder_loss
    
    assert torch.allclose(total_loss, expected_total)
    
    # Check that metrics are captured
    metrics = loss_fn.get_metrics()
    assert torch.allclose(metrics["ce_loss"], expected_ce)
    assert torch.allclose(metrics["ponder_loss"], ponder_loss)
    assert metrics["metrics"]["some_metric"].item() == pytest.approx(4.56)

