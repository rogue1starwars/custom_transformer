import copy
from torch import nn

from model import transformer_model


def build_model(
    src_vocab: int,
    tgt_vocab: int,
    N=6,
    d_model=512,
    d_ff=2048,
    h=8,
    dropout=0.1,
) -> nn.Module:
    """Build a Transformer model.

    Args:
        src_vocab (int): Size of the source vocabulary.
        tgt_vocab (int): Size of the target vocabulary.
        N (int): Number of layers in the encoder and decoder.
        d_model (int): Dimension of the model.
        d_ff (int): Dimension of the feedforward network.
        h (int): Number of heads in multi-head attention.
        dropout (float): Dropout rate.

    Returns:
        nn.Module: The Transformer model.
    """
    c = copy.deepcopy
    attn = transformer_model.MultiHeadAttention(h, d_model)
    ff = transformer_model.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = transformer_model.PositionalEncoding(d_model, dropout)
    model = transformer_model.EncoderDecoder(
        encoder=transformer_model.Encoder(
            transformer_model.EncoderLayer(d_model, c(attn), c(ff), dropout), N
        ),
        decoder=transformer_model.Decoder(
            transformer_model.DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N
        ),
        src_embed=nn.Sequential(
            transformer_model.Embeddings(d_model, src_vocab), c(position)
        ),
        tgt_embed=nn.Sequential(
            transformer_model.Embeddings(d_model, tgt_vocab), c(position)
        ),
        generator=transformer_model.Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
