from typing import Tuple

import torch
from torch import nn, Tensor

from utils import helper_functions


class Batch:
    """Batch object to hold source and target sequences along with their masks.

    This class is used to prepare the data for the Transformer model.

    Attributes:
        src (Tensor): The source sequence tensor, shape (batch_size, seq_length).
        tgt (Tensor, optional): The target sequence tensor, shape (batch_size, seq_length).
        src_mask (Tensor): The mask for the source sequence, shape (batch_size, 1, seq_length).
        tgt_mask (Tensor, optional): The mask for the target sequence, shape (batch_size, seq_len, seq_len).
        tgt_y (Tensor, optional): The target sequence tensor shifted by one position, shape (batch_size, seq_length - 1).
        ntokens (int, optional): The number of tokens in the target sequence, used for normalization.
    """

    def __init__(self, src, tgt=None, pad=2):
        """Initialize the Batch object.
        Args:
            src (Tensor):
                The source sequence tensor. Shape should be (batch_size, seq_length).
            tgt (Tensor, optional):
                The target sequence tensor. If not provided, only the source sequence is used.
                The shape should be (batch_size, seq_length).
            pad (int):
                The padding token index, default is 2 which is <blank>.
        """
        # src shape: (batch_size, token sequence length)
        self.src = src
        self.src_mask = (src != pad).unsqueeze(
            -2
        )  # adding another dimention -> (batch_size, extra dim, token sequence length)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a standard mask for the target sequence.
        This mask prevents the model from attending to padding tokens and
        future tokens in the sequence.

        Args:
            tgt (Tensor): The target sequence tensor, shape (batch_size, seq_length).
            pad (int): The padding token index.
        Returns:
            Tensor: A mask tensor of shape (batch_size, seq_len, seq_len) where positions that are
            allowed to attend to future positions are set to 0, and all other positions are set to 1.
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & helper_functions.create_subsequent_mask(
            tgt.size(-1)
        ).type_as(tgt_mask.data)
        return tgt_mask


def rate(
    step: int,
    model_size: int,
    factor: float,
    warmup: float,
) -> float:
    """Function to calculate Learning Rate

    The equation is as follows:
    Lr = factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

    Note: Step must be bigger than 0 to avoid zero raising to negative power.

    Args:
        step (int): The current step in training.
        model_size (int): The size of the model, typically the dimension of the embeddings.
        factor (float): A scaling factor for the learning rate.
        warmup (float): A warmup factor to control the learning rate increase.

    Returns:
        float: The calculated learning rate.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class SimpleLossCompute:
    """A simple loss compute for training and evaluation

    This class computes the loss for a given input and target using a generator
    and a criterion. It normalizes the loss by a given norm value.

    Attributes:
        generator (nn.Module): A generator module that transforms the input.
        criterion (nn.Module): A loss criterion to compute the loss.
    """

    def __init__(self, generator: nn.Module, criterion: nn.Module):
        """Initialize the SimpleLossCompute with a generator and criterion.

        Args:
            generator (nn.Module): The generator module to transform the input.
            criterion (nn.Module): The loss criterion to compute the loss.
        """
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: Tensor, y: Tensor, norm: int) -> Tuple[Tensor, Tensor]:
        """Compute the loss for the given input and target.

        Args:
            x (torch.Tensor):
                The input tensor, typically the output from the model.
                The shape should be (batch_size, seq_length, vocab_size).
            y (torch.Tensor):
                The target tensor, typically the ground truth labels.
            norm (int):
                A normalization factor, usually the number of tokens.

        Returns:
            tuple: A tuple containing the normalized loss and the raw loss.
        """
        x = self.generator(x)
        sloss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        return sloss.data * norm, sloss


class LabelSmoothing(nn.Module):
    """Label Smoothing Loss

    This class implements label smoothing, which is a technique to prevent overfitting
    by softening the target labels. It is particularly useful in classification tasks.

    Attributes:
        criterion (nn.Module): The loss criterion, typically KLDivLoss.
        padding_idx (int): The index of the padding token in the vocabulary.
        confidence (float): The confidence level for the true class, typically 1 - smoothing.
        smoothing (float): The smoothing factor, typically between 0 and 1.
        size (int): The size of the vocabulary.
        true_dist (torch.Tensor): The true distribution of labels after smoothing.
        It is initialized to None and computed during the forward pass.
    """

    def __init__(self, size: int, padding_idx, smoothing=0.0):
        """Initialize the LabelSmoothing loss.

        Args:
            size (int): The size of the vocabulary.
            padding_idx (int): The index of the padding token.
            smoothing (float): The smoothing factor, typically between 0 and 1.
        """
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """Compute the label smoothed loss.

        Args:
            x (torch.Tensor): The model output, shape (batch_size * seq_length, vocab_size).
            target (torch.Tensor): The target labels, shape (batch_size * seq_length).
            The target labels should contain the indices of the true classes.

        Returns:
            torch.Tensor: The computed loss value.
        """
        assert x.size(1) == self.size
        true_dist = (
            x.detach().clone()
        )  # Clone the input tensor to create a true distribution
        true_dist.fill_(self.smoothing / (self.size - 2))  # Fill with smoothing value
        true_dist.scatter_(
            1, target.data.unsqueeze(1), self.confidence
        )  # Set the true class confidence
        # Set the padding index to zero in the true distribution
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(
                0, mask.squeeze(), 0.0
            )  # Set the true distribution to zero for padding indices
        self.true_dist = (
            true_dist  # Store the true distribution for potential future use
        )
        return self.criterion(x, true_dist.clone().detach())


class DummyOptimizer(torch.optim.Optimizer):
    """A dummy optimizer that does nothing."""

    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    """A dummy scheduler that does nothing."""

    def step(self):
        None
