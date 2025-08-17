import torch
from torch import nn, Tensor


def create_subsequent_mask(size):
    """Mask out subsequent positions in a sequence.

    This function creates a mask that prevents the decoder from
    attending to future tokens in the sequence.
    The mask is a square matrix of shape (1, size, size) where
    positions that are allowed to attend to future positions
    are set to 0, and all other positions are set to 1.

    Args:
        size (int): The size of the sequence for which the mask is created.
    Returns:
        Tensor: A boolean mask of shape (1, size, size) where True indicates
        positions that can be attended to, and False indicates positions that cannot be attended to.
        Example:
        >>> mask = create_subsequent_mask(5)
        >>> print(mask)
        tensor([[[ True, False, False, False, False],
                 [ True,  True, False, False, False],
                 [ True,  True,  True, False, False],
                 [ True,  True,  True,  True, False],
                 [ True,  True,  True,  True,  True]]], dtype=torch.uint8)
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def greedy_decode(
    model: nn.Module,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    start_symbol: int,
) -> Tensor:
    """Decodes the input sequence using greedy decoding.

    This function generates a sequence by iteratively predicting the next token
    based on the previous tokens and the model's output.

    Args:
        model (nn.Module): The trained Transformer model.
        src (torch.Tensor): The source sequence tensor, shape (batch_size, seq_length).
        src_mask (torch.Tensor): The mask for the source sequence, shape (batch_size, seq_length).
        max_len (int): The maximum length of the output sequence to generate.
        start_symbol (int): The index of the start token in the vocabulary.

    Returns:
        torch.Tensor: The generated sequence tensor, shape (1, max_len).
    """
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory,
            src_mask,
            ys,
            create_subsequent_mask(ys.size(1)).type_as(src.data),
        )
        prob = model.generator(out[:, -1])  # printing out the prob for the last output
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    return ys
