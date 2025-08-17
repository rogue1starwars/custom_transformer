import math
import copy
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class EncoderDecoder(nn.Module):
    """A Transformer model that consists of an encoder and a decoder.

    Encoder processes the source sequence and decoder generates the target sequence.
    The model uses embeddings for both source and target sequences, and a generator
    to produce the final output logits.

    Attributes:
        encoder (nn.Module):
            The encoder part of the Transformer.
            Consists of multiple layers of self-attention and feed-forward networks.
        decoder (nn.Module):
            The decoder part of the Transformer.
            Similar to the encoder but also attends to the encoder's output.
        src_embed (nn.Module):
            Embedding layer for the source sequence.
        tgt_embed (nn.Module):
            Embedding layer for the target sequence.
        generator (nn.Module):
            A linear layer followed by a softmax to generate the final output logits.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: nn.Module,
    ):
        """Initialize the Transformer model.

        Args:
            encoder (nn.Module): The encoder module.
            decoder (nn.Module): The decoder module.
            src_embed (nn.Module): The embedding layer for the source sequence.
            tgt_embed (nn.Module): The embedding layer for the target sequence.
            generator (nn.Module): The generator module to produce final output logits.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """Forward pass through the Transformer model.

        Args:
            src (Tensor):
                The source sequence tensor. The shape is (batch_size, src_len).
            tgt (Tensor):
                The target sequence tensor. The shape is (batch_size, tgt_len).
                This is the ground truth for training. During inference, it receives
                the previously generated symbols as additional input.
            src_mask (Tensor):
                The source sequence mask. The shape is (batch_size, 1, 1, src_len).
            tgt_mask (Tensor):
                The target sequence mask. The shape is (batch_size, 1, tgt_len, tgt_len).

        Returns:
            Tensor:
                A sequence of vector representations for the target sequence.
                The shape is (batch_size, tgt_len, d_model).
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: nn.Module, src_mask: nn.Module) -> Tensor:
        """Encode the source sequence.

        This method processes receives the source symbol representation sequence and its mask,
        applies the source embeddings to convert each token into a vector representation,
        and passes it through the encoder to produce a memory representation.

        Args:
            src (Tensor): The source sequence tensor. The shape is (batch_size, src_len).
            src_mask (Tensor): The source sequence mask. The shape is (batch_size, 1, 1, src_len).

        Returns:
            Tensor:
                A meomory representation of the source sequence.
                The shape is (batch_size, src_len, d_model).
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: nn.Module,
        src_mask: nn.Module,
        tgt: nn.Module,
        tgt_mask: nn.Module,
    ) -> nn.Module:
        """Decode the target sequence.

        This method receives the memory representation from the encoder, the source mask,
        the target sequence, and the target mask. It applies the target embeddings to
        convert each token into a vector representation, and passes it through the decoder

        Args:
            memory (Tensor):
                The memory representation from the encoder.
                The shape is (batch_size, src_len, d_model).
            src_mask (Tensor):
                The source sequence mask. The shape is (batch_size, 1, 1, src_len).
            tgt (Tensor):
                The target sequence tensor. The shape is (batch_size, tgt_len).
                This is the ground truth for training. During inference, it receives
                the previously generated symbols as additional input.
            tgt_mask (Tensor):
                The target sequence mask. The shape is (batch_size, 1, tgt_len, tgt_len).

        Returns:
            Tensor:
                A sequence of vector representations for the target sequence.
                The shape is (batch_size, tgt_len, d_model).
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Generate the final vocab using linear layer and softmax

    This module takes the output from the decoder and projects it to the vocabulary size
    using a linear layer, followed by a log softmax to produce probabilities for each token in
    the vocabulary.

    Attributes:
        proj (nn.Linear): A linear layer that projects the decoder output to the vocabulary size.
    """

    def __init__(self, d_model, vocab):
        """Initialize the Generator module with the given model dimension and vocabulary size.

        Args:
            d_model (int): The dimension of the model (the size of the output from the decoder).
            vocab (int): The size of the vocabulary (the number of unique tokens).
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """Forward pass through the Generator module.

        This method applies the linear projection to the input tensor and then applies
        the log softmax function to produce probabilities for each token in the vocabulary.
        The softmax function is applied along the last dimension (-1),
        which corresponds to the vocabulary size.

        Args:
            x (Tensor): The input tensor from the decoder.
            The shape is (batch_size, tgt_len, d_model).

        Returns:
            Tensor:
                The output tensor containing log probabilities for each token in the vocabulary.
                The shape is (batch_size, tgt_len, vocab).
        """
        return F.log_softmax(self.proj(x), dim=-1)


#
# Encoder and Decoder
#


class Encoder(nn.Module):
    """Encoder module for the Transformer model.

    This module consists of multiple identical layers, each containing a self-attention mechanism.

    Attributes:
        layers (nn.ModuleList): A list of identical encoder layers.
        norm (nn.LayerNorm): A layer normalization applied to the output of the encoder.
    """

    def __init__(self, layer: nn.Module, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Forward pass through the Encoder module.

        Args:
            x (Tensor):
                The input tensor to the encoder. The shape is (batch_size, src_len, d_model).
            mask (Tensor):
                The mask for the input sequence. The shape is (batch_size, 1, 1, src_len).
                Note:
                    The mask is used to prevent attention to padding tokens in the source sequence.
                    Padding tokens are typically added to ensure that all sequences in a batch
                    have the same length.

                    Example.
                        Suppose the sequence length are fixed to 5.
                        src = ['I', 'love', 'programming']
                        will be padded to:
                        src = ['I', 'love', 'programming', <pad>, <pad>]
                        where <pad> is a padding token.
                        The mask will look like this:
                        mask = [[[True, True, True, False, False]]]
                        where True indicates a valid token and False indicates a padding token.

                    The model should not attend to the padding tokens, as they do not
                    contain any useful information.

        Returns:
            Tensor:
                The output tensor after passing through all encoder layers and normalization.
                The shape is (batch_size, src_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder layer consisting of self-attention and feed-forward sublayers.

    The EncoderLayer applies self-attention to the input sequence and then passes the result
    through a feed-forward network. Each sublayer is wrapped in a SublayerConnection to
    facilitate residual connections and layer normalization.

    Attributes:
        self_attn (nn.Module): The multi-head self-attention mechanism.
        feed_forward (nn.Module): The feed-forward network.
        attention_layer (SublayerConnection): The sublayer connection for the self-attention mechanism.
        feed_forward_layer (SublayerConnection): The sublayer connection for the feed-forward network.
        size (int): The size of the input and output tensors for this layer.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float,
    ):
        """Initialize the EncoderLayer with self-attention and feed-forward networks.

        Args:
            size (int): The size of the input and output tensors for this layer.
            self_attn (nn.Module): The multi head self attention mechanism.
            feed_forward (nn.Module): The feed-forward network.
            dropout_rate (float): The dropout rate to apply after the sublayers.
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.attention_layer = SublayerConnection(size, dropout_rate)
        self.feed_forward_layer = SublayerConnection(size, dropout_rate)
        self.size = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Forward pass through the EncoderLayer.

        Note:
            The SublayerConnection accepts a callable (function) as the second argument.
            Here, we use a lambda function to pass the self-attention mechanism.
            x in lambda x: self.self_attn(x, x, x, mask) is different from x in the first argument.

        Args:
            x (Tensor):
                The input tensor to the encoder layer. The shape is (batch_size, src_len, d_model).
            mask (Tensor):
                The mask for the input src sequence. The shape is (batch_size, 1, 1, src_len).

        Returns:
            Tensor:
                The output tensor after applying multi-head self-attention and feed-forward networks.
                The shape is (batch_size, src_len, d_model).
        """

        x = self.attention_layer(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.feed_forward_layer(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """Decoder module for the Transformer model.

    This module consists of multiple identical layers, each containing a self-attention mechanism
    and a cross-attention mechanism that attends to the encoder's output.

    Attributes:
        layers (nn.ModuleList): A list of identical decoder layers.
        norm (nn.LayerNorm): A layer normalization applied to the output of the decoder.
    """

    def __init__(self, layer: nn.Module, N: int):
        """Initialize the Decoder module with the given layer and number of layers.

        Args:
            layer (nn.Module): The decoder layer to be cloned.
            N (int): The number of identical decoder layers to create.
        """
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """Forward pass through the Decoder module.

        This method processes the target sequence tensor, applies the target embeddings,
        and passes it through the decoder layers. Each layer applies self-attention and
        cross-attention to the memory representation from the encoder.

        Args:
            x (Tensor):
                The input tensor to the decoder.
                The shape is (batch_size, tgt_len, d_model).
            memory (Tensor):
                The memory representation from the encoder.
                The shape is (batch_size, src_len, d_model).
            src_mask (Tensor):
                The source sequence mask.
                The shape is (batch_size, 1, 1, src_len).
                Look at the Encoder's forward method for more details.
            tgt_mask (Tensor):
                The target sequence mask.
                The shape is (batch_size, 1, tgt_len, tgt_len).

                Note:
                    The tgt mask is used to prevent attention to the future tokens in the target sequence.
                    When training, the model receives the entire target sequence (ground truth) as input.
                    We apply a mask in order to prevent the model from cheating.
                    This is also useful during inference, where the model generates the target sequence
                    one token at a time, and we want to prevent it from attending to future tokens
                    since they are not available yet.

        Returns:
            Tensor:
                The output tensor after passing through all decoder layers and normalization.
                The shape is (batch_size, tgt_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder layer consisting of self-attention, source attention, and feed-forward sublayers.

    This layer applies self-attention to the target sequence, attends to the source sequence,
    and then passes the result through a feed-forward network. Each sublayer is wrapped in a
    SublayerConnection to facilitate residual connections and layer normalization.

    Attributes:
        size (int): The size of the input and output tensors for this layer.
        self_attn (nn.Module): The multi-head self-attention mechanism for the target sequence.
        src_attn (nn.Module): The multi-head attention mechanism that attends to the source sequence.
        feed_forward (nn.Module): The feed-forward network.
        self_attn_layer (SublayerConnection): The sublayer connection for the self-attnention mechanism.
        src_attn_layer (SublayerConnection): The sublayer connection for the source attention mechanism.
        feed_forward_layer (SublayerConnection): The sublayer connection for the feed-forward network.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float,
    ):
        """Initialize the DecoderLayer with self-attention, source attention, and feed-forward networks.

        Args:
            size (int): The size of the input and output tensors for this layer.
            self_attn (nn.Module): The multi-head self-attention mechanism for the target sequence.
            src_attn (nn.Module): The multi-head attention mechanism that attends to the source sequence.
            feed_forward (nn.Module): The feed-forward network.
            dropout (float): The dropout rate to apply after the sublayers.
        """
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.self_attn_layer = SublayerConnection(size, dropout_rate)
        self.src_attn_layer = SublayerConnection(size, dropout_rate)
        self.feed_forward_layer = SublayerConnection(size, dropout_rate)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """Forward pass through the DecoderLayer.

        Args:
            x (Tensor):
                The input tensor to the decoder layer.
                The shape is (batch_size, tgt_len, d_model).
            memory (Tensor):
                The memory representation from the encoder.
                The shape is (batch_size, src_len, d_model).
            src_mask (Tensor):
                The source sequence mask.
                The shape is (batch_size, 1, 1, src_len).
                Look at the Encoder's forward method for more details.
            tgt_mask (Tensor):
                The target sequence mask.
                The shape is (batch_size, 1, tgt_len, tgt_len).
                Look at the Decoder's forward method for more details.

        Returns:
            Tensor: The output tensor after passing through the decoder layer.
            The shape is (batch_size, tgt_len, d_model).
        """
        m = memory
        x = self.self_attn_layer(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.src_attn_layer(x, lambda x: self.src_attn(x, m, m, src_mask))
        x = self.feed_forward_layer(x, self.feed_forward)
        return x


class SublayerConnection(nn.Module):
    """Sublayer connection with residual connection and layer normalization."""

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: callable) -> Tensor:
        """Forward passes through a sublayer connection.

        This method applies layer normalization to the input tensor, passes it through the sublayer,
        and then adds the original input tensor back to the output of the sublayer.

        Note:
            Different from the original paper, we apply layer normalization before the sublayer.
            Dropout is applied to the output of the sublayer before adding the residual connection.
            (Mentioned in Chapter 5.4 Regularization)

        Args:
            x (Tensor):
                The input tensor to the sublayer connection.
            sublayer (callable):
                A callable that takes the normalized input tensor and returns the output tensor.

        Returns:
            Tensor:
                The output tensor after applying the sublayer and adding the residual connection.
                The shape is the same as the input tensor (batch_size, seq_len, d_model).
        """
        return x + self.dropout(sublayer(self.norm(x)))


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute Scaled Dot Product Attention

    This function computes the attention scores between the query and key tensors,
    applies a mask if provided, and then computes the weighted sum of the value tensor.

    Args:
        query (Tensor): The query tensor of shape (batch_size, num_heads, seq_len, head_dim).
        key (Tensor): The key tensor of shape (batch_size, num_heads, seq_len, head_dim).
        value (Tensor): The value tensor of shape (batch_size, num_heads, seq_len, head_dim).
        mask (Tensor, optional): A mask tensor of shape (batch_size, 1, 1 or (seq_len), seq_len) to prevent attention to certain
            positions. If provided, positions where the mask is 0 will not be attended to.
        dropout (nn.Module, optional): A dropout layer to apply to the attention probabilities.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - The output tensor of shape (batch_size, num_heads, seq_len, head_dim).
            - The attention probabilities tensor of shape (batch_size, num_heads, seq_len, seq_len).
    """
    d_k = query.size(-1)

    # Query, key, value shape: (batch_size, num_heads, seq_len, head_dim)
    # Transpose the last 2 dimensions. The shape after transpose will be (batch_size, num_heads, head_dim, seq_len)
    # The shape after matmul will be (batch_size, num_heads, seq_len, seq_len)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # The shape after matmul will be (batch_size, num_heads, seq_len, head_dim)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """Multi-Headed Attention module.

    This module implements the multi-headed attention mechanism as described in the Transformer paper.
    It consists of multiple linear layers for query, key, and value projections,
    and a final linear layer to combine the outputs of the attention heads.

    Attributes:
        head_num (int): The number of attention heads.
        head_dimension (int): The dimension of each attention head.
        linears (nn.ModuleList): A list of linear layers for query, key, and value projections.
        out_projection_net (nn.Linear): A linear layer to project the concatenated outputs
            of the attention heads back to the model dimension.
        attn (Tensor, optional): The attention weights computed during the forward pass.
        dropout (nn.Dropout): A dropout layer to apply to the attention probabilities.
    """

    def __init__(self, head_num: int, d_model: int, dropout_rate: float = 0.1):
        """Initialize the MultiHeadAttention module.

        Args:
            head_num (int): The number of attention heads.
            d_model (int): The dimension of the model (the size of the input and output tensors).
            dropout (float): The dropout rate to apply to the attention probabilities.
        """
        super().__init__()
        assert (
            d_model % head_num == 0
        )  # Ensure that the model dimension is divisible by the number of heads

        self.head_dimension = int(
            d_model / head_num
        )  # The dimension of each attention head.
        self.head_num = head_num
        self.linears = clones(
            nn.Linear(d_model, d_model), 3
        )  # Create 3 identical linear layers for query, key, and value projections
        self.out_projection_net = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        input_query: Tensor,
        input_key: Tensor,
        input_value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the MultiHeadAttention module.

        This method applies the multi-headed attention mechanism to the input tensors.
        Note:
            - The input tensors (batch_size, seq_len, d_model) will be projected
            to shape (batch_size, seq_len, head_num, head_dimension) for each of the query, key, and value tensors.
            - The attention mechanism will compute the attention scores and apply the mask if provided.


        Args:
            input_query (Tensor): The query tensor of shape (batch_size, seq_len, d_model).
            input_key (Tensor): The key tensor of shape (batch_size, seq_len, d_model).
            input_value (Tensor): The value tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): A mask tensor of shape (batch_size, 1, 1 or (seq_len), seq_len)
                to prevent attention to certain positions. If provided, positions where
                the mask is False will not be attended to.

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, d_model).
            The output tensor is the result of applying the multi-headed attention mechanism
            to the input tensors, followed by a linear projection to the model dimension.
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = input_query.size(0)

        # Linearly project the query, key, and value tensors.
        # First, the linear layers will project each tensors into shape (batch_size, seq_len, d_model).
        # Then, we reshape them to (batch_size, seq_len, head_num, head_dimension)
        # Finally, we transpose the last two dimensions to get (batch_size, head_num, seq_len, head_dimension)
        query, key, value = [
            net(x)  # (batch_size, seq_len, d_model)
            .view(
                batch_size, -1, self.head_num, self.head_dimension
            )  # (batch_size, seq_len, head_num, head_dimension)
            .transpose(1, 2)  # (batch_size, head_num, seq_len, head_dimension)
            for net, x in zip(self.linears, (input_query, input_key, input_value))
        ]

        # The output tensor will be of shape (batch_size, head_num, seq_len, head_dimension)
        # The attention mechanism will compute the attention scores and apply the mask if provided.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate the outputs of all attention heads and reshape to (batch_size, seq_len, d_model)
        # First, we transpose the 2nd and 3rd dimension (indexed as 1, 2) in the middle
        # to get (batch_size, seq_len, head_num, head_dimension)
        # Then, we use contiguous to ensure the tensor is contiguous in memory
        # Finally, we reshape it to (batch_size, seq_len, head_num * head_dimension)
        # This is the final output tensor that will be passed to the next layer.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.head_num * self.head_dimension)
        )

        del query
        del key
        del value

        # Apply the final linear projection to the concatenated outputs of all attention heads
        # The output tensor will be of shape (batch_size, seq_len, d_model)
        return self.out_projection_net(x)


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network.

    In addition to the muti-headed attention sublayers, each layer in the encoder and decoder contains a
    fully connected feed-forward network.

    Attributes:
        w_1 (nn.Module): A feed forward network that projects the input to a higher dimension.
        w_2 (nn.Module): A feed forward network that projects the input back to the original dimension.
        dropout (nn.Dropout): A dropout layer to apply after the first linear transformation
    """

    def __init__(self, d_model: int, d_feed_forward: int, dropout_rate: float = 0.1):
        """Initialize the PositionwiseFeedForward module.

        Args:
            d_model (int): The dimension of the input and output tensors.
            d_feed_forward (int): The dimension of the hidden layer in the feed-forward network.
            dropout_rate (float): The dropout rate to apply after the first linear transformation.
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_feed_forward)
        self.w_2 = nn.Linear(d_feed_forward, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the PositionwiseFeedForward module.

        This method applies a two-layer feed-forward network to the input tensor.

        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.w_2(self.dropout(self.w_1(x).relu()))


#
# Input Embedding and Positional Encoding
#


class Embeddings(nn.Module):
    """Embeddings module for the Transformer model."""

    def __init__(self, d_model, vocab):
        """Initialize the Embeddings module.

        Args:
            d_model (int): The dimension of the model (the size of the output embeddings).
            vocab (int): The size of the vocabulary (the number of unique tokens).
        """
        super(Embeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """Forward pass through the Embeddings module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_len).

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        output = self.embeddings(x) * math.sqrt(self.d_model)
        return output


class PositionalEncoding(nn.Module):
    """Positional Encoding module for the Transformer model."""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Initialize the PositionalEncoding module.

        Equations:
            PE(pos, 2i) = sin(pos/10000^(2i/d_model))
            PE(pos, 2i+1) = sin(pos/10000^(2i/d_model))

            'i' corresponds to each dimension (from 0 to d_model) and 'pos' corresponds to the
            position in the token sequence

        Args:
            d_model (int): The dimension of the model (the size of the output embeddings).
            dropout_rate (float): The dropout rate to apply after the positional encoding.
            max_len (int): The maximum length of the input sequence for which positional encodings are created.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        position = torch.arange(0, max_len).unsqueeze(1)

        frequencies = torch.pow(
            10000.0, -torch.arange(0, d_model, 2, dtype=torch.float) / d_model
        )

        positional_encodings_table = torch.zeros(max_len, d_model)
        positional_encodings_table[:, 0::2] = torch.sin(position * frequencies)
        positional_encodings_table[:, 1::2] = torch.cos(position * frequencies)

        self.register_buffer("positional_encoding_table", positional_encodings_table)

    def forward(self, embeddings_batch: Tensor) -> Tensor:
        """Forward pass through the PositionalEncoding module.

        Args:
            embeddings_batch (Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, d_model) with positional encodings added.
        """
        positional_encodings = self.positional_encoding_table[
            : embeddings_batch.shape[1]
        ]
        return self.dropout(embeddings_batch + positional_encodings)


#
# Helper Functions
#


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Helper function: Produce N identical Layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
