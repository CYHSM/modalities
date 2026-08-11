import torch

from modalities.batch import DatasetBatch
from modalities.dataloader.collate_fns.collate_if import CollateFnIF


class ExplicitTargetCollateFn(CollateFnIF):
    """Collate function for datasets that emit their own, already shifted, targets.

    ``GPT2LLMCollateFn`` derives the targets from the samples by shifting, which makes every
    position a training signal. Evaluation datasets that ask a single question per sample -- see
    :mod:`modalities.dataloader.synthetic_reasoning` -- instead need targets that are masked
    everywhere except at the answer, so they supply the targets themselves and this collator
    passes them through unchanged.
    """

    def __init__(self, sample_key: str, target_key: str):
        """
        Initializes the collator.

        Args:
            sample_key (str): The key under which the dataset emits the input token ids.
            target_key (str): The key under which the dataset emits the targets.
        """
        self.sample_key = sample_key
        self.target_key = target_key

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> DatasetBatch:
        """
        Stacks a list of samples into a batch without shifting.

        Args:
            batch (list[dict[str, torch.Tensor]]): Samples, each carrying both keys.

        Returns:
            DatasetBatch: The batched samples and targets, of equal shape.
        """
        samples = torch.stack([torch.as_tensor(sample[self.sample_key]) for sample in batch])
        targets = torch.stack([torch.as_tensor(sample[self.target_key]) for sample in batch])
        return DatasetBatch(samples={self.sample_key: samples}, targets={self.target_key: targets})
