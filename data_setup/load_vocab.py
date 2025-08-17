import os
from collections import Counter

import pandas as pd

from data_setup import data_utils


def build_manual_vocab(
    token_iter: iter,
    min_freq: int = 1,
    specials: list = None,
    special_first: bool = True,
    chunk_size: int = 10000,
) -> data_utils.Vocab:
    """Build a vocabulary from an iterable of tokens.
    This function counts the frequency of each token and constructs a vocabulary
    with tokens that appear at least `min_freq` times. It also allows for special tokens
    to be added at the beginning or end of the vocabulary.

    Args:
        token_iter (iter): An iterable that yields tokens.
        min_freq (int): Minimum frequency for a token to be included in the vocabulary.
        specials (list): A list of special tokens to include in the vocabulary.
        special_first (bool): If True, special tokens are added at the beginning of the vocab.
        chunk_size (int): Number of tokens to process at a time to avoid memory issues.

    Returns:
        Vocab: An instance of the Vocab class containing the built vocabulary.
    """
    counter = Counter()
    # Process tokens in chunks to avoid memory issues
    chunk = []
    for token in token_iter:
        chunk.append(token)
        if len(chunk) == chunk_size:
            counter.update(chunk)
            chunk = []
    if chunk:  # Process any remaining tokens
        counter.update(chunk)

    print("Finished counting. Filtering tokens...")

    # Filter tokens below min_freq
    tokens_and_counts = [(tok, cnt) for tok, cnt in counter.items() if cnt >= min_freq]
    # Sort by frequency (desc), then alphabetically
    tokens_and_counts.sort(key=lambda x: (-x[1], x[0]))

    # Order tokens
    vocab_tokens = [tok for tok, _ in tokens_and_counts]
    if specials:
        if special_first:
            vocab_tokens = specials + vocab_tokens
        else:
            vocab_tokens = vocab_tokens + specials

    # Build stoi and itos
    itos = vocab_tokens
    stoi = {tok: idx for idx, tok in enumerate(itos)}

    # Default OOV index
    default_index = stoi.get("<unk>", None)

    v = data_utils.Vocab(stoi, itos, default_index)
    return v


def build_vocabulary(
    language: str, tokenizer: callable, raw_df: pd.DataFrame
) -> data_utils.Vocab:
    """Build a vocabulary for a specific language using a tokenizer.

    This function tokenizes the sentences in the specified language and builds a vocabulary
    using the `build_manual_vocab` function. It yields tokens from the tokenized sentences.

    Args:
        language (str): The language for which the vocabulary is built (e.g., "Japanese").
        tokenizer (callable): A tokenizer function that takes a sentence and returns a list of tokens.
        raw_df (pd.DataFrame): A DataFrame containing the raw data with sentences in the specified language.

    Returns:
        Vocab: An instance of the Vocab class containing the built vocabulary.
    """
    print(f"Building {language} Vocab...")
    vocab_raw = raw_df[language]
    vocab = build_manual_vocab(
        yield_tokens(vocab_raw, tokenizer),
        min_freq=1,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    return vocab


def load_vocab(spacy_ja, spacy_en, raw_df, root_path="./vocabs"):
    """Load or build the vocabulary for both source and target languages.

    This function checks if the vocabulary files exist in the specified directory.
    If they do, it loads them; otherwise, it builds the vocabularies from the raw data.

    Args:
        spacy_ja: The Japanese tokenizer.
        spacy_en: The English tokenizer.
        raw_df (pd.DataFrame): The DataFrame containing the raw data with sentences in both languages.
        root_path (str): The directory where the vocabulary files are stored.

    Returns:
        tuple: A tuple containing the source and target vocabularies as Vocab instances.
    """
    src_path = os.path.join(root_path, "vocab_src.csv")
    tgt_path = os.path.join(root_path, "vocab_tgt.csv")
    if not os.path.exists(src_path):
        print("Building Japanese Vocabulary...")
        vocab_src = build_vocabulary("Japanese", spacy_ja, raw_df)
        vocab_src.to_csv(src_path)
    else:
        vocab_src = data_utils.Vocab.from_csv(src_path)
    if not os.path.exists(tgt_path):
        print("Building English Vocabulary...")
        vocab_tgt = build_vocabulary("English", spacy_en, raw_df)
        vocab_tgt.to_csv(tgt_path)
    else:
        vocab_tgt = data_utils.Vocab.from_csv(tgt_path)
    print("Finished.\nVocabulary sizes: ")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


#
# Helper functions for tokenization
#


def tokenize(text, tokenizer):
    """Tokenize a single text using the provided tokenizer."""
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer):
    """Yield tokens from the data iterator using the specified tokenizer."""
    for sentence in data_iter:
        yield from tokenize(sentence, tokenizer)
