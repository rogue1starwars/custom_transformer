import tarfile
import re
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from data_setup import data_utils


def extract_data(root_path="./data"):
    """Extracts the raw data from the tar.gz file and prepares it for use.

    This function downloads the raw data file, extracts it, and processes it into a Pandas DataFrame.
    The DataFrame contains two columns: 'English' and 'Japanese', representing the English and Japanese sentences from the dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the English and Japanese sentences from the dataset.
    """
    path = os.path.join(root_path, "raw.tar.gz")
    savedir = "./data_raw/"
    filename = "raw"
    save_path = os.path.join(savedir + "raw/", filename)
    try:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if not os.path.exists(path):
            print(f"Downloading raw data to {path}...")
            os.system(
                f"wget -P {root_path} -nc https://nlp.stanford.edu/projects/jesc/data/raw.tar.gz"
            )
        else:
            print(f"Raw data already exists at {path}, skipping download.")

        if not os.path.exists(save_path):
            with tarfile.open(path, "r:*") as tar:
                tar.extractall(savedir)
        else:
            print(f"Raw data already extracted at {save_path}, skipping extraction.")

        with open(os.path.join(savedir + "raw/", filename), "r", encoding="utf-8") as f:
            raw_data = f.readlines()

    except (OSError, IOError, tarfile.TarError) as e:
        print(f"Error extracting data: {e}")
        return pd.DataFrame(columns=["English", "Japanese"])

    raw_list = [re.sub("\n", "", s).split("\t") for s in raw_data]
    raw_df = pd.DataFrame(raw_list, columns=["English", "Japanese"])
    raw_df = raw_df.dropna()
    raw_df = raw_df[raw_df["English"] != ""]
    raw_df = raw_df[raw_df["Japanese"] != ""]
    return raw_df


class TranslationDataset(Dataset):
    """Dataset class for translation tasks."""

    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        return self.df.iloc[idx]["Japanese"], self.df.iloc[idx]["English"]

    def __len__(self):
        return len(self.df)


def create_dataloaders(
    raw_df: pd.DataFrame,
    device: torch.device,
    vocab_src: data_utils.Vocab,
    vocab_tgt: data_utils.Vocab,
    spacy_ja: object,
    spacy_en: object,
    batch_size: int,
    max_padding: int = 128,
):
    """Create DataLoader for the translation dataset.

    This function prepares the dataset and returns a DataLoader for training or evaluation.

    Args:
        root_path (str): The root path where the raw data is stored.
        device (torch.device): The device to run the model on (e.g., "cpu" or "cuda").
        vocab_src (Vocab): The source vocabulary.
        vocab_tgt (Vocab): The target vocabulary.
        spacy_ja: The Japanese tokenizer.
        spacy_en: The English tokenizer.
        batch_size (int): The batch size for the DataLoader.
        max_padding (int): The maximum padding length for sequences.
    Returns:
        DataLoader: A DataLoader instance for the translation dataset.
    """

    def tokenize(text, tokenizer):
        """Tokenize a single text using the provided tokenizer."""
        return [tok.text for tok in tokenizer.tokenizer(text)]

    def tokenize_ja(text):
        """Tokenize Japanese text using the provided tokenizer."""
        return tokenize(text, spacy_ja)

    def tokenize_en(text):
        """Tokenize English text using the provided tokenizer."""
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        """Collate function to prepare batches of data."""
        return collate_batch(
            batch,
            tokenize_ja,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    if raw_df.empty:
        print("No data found. Returning empty DataLoader.")
        return DataLoader([], batch_size=batch_size, collate_fn=collate_fn)
    dataset = TranslationDataset(raw_df)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )


def collate_batch(
    batch: list,
    src_pipeline: callable,
    tgt_pipeline: callable,
    src_vocab: data_utils.Vocab,
    tgt_vocab: data_utils.Vocab,
    device: torch.device,
    max_padding: int = 128,
    pad_id: int = 2,
) -> tuple:
    """Collate function to prepare batches of data for the DataLoader.

    This function processes the source and target sequences, applies padding, and converts them to tensors.

    Args:
        batch (list): A list of tuples containing source and target sentences.
        src_pipeline (callable): A function to tokenize the source sentences.
        tgt_pipeline (callable): A function to tokenize the target sentences.
        src_vocab (Vocab): The source vocabulary.
        tgt_vocab (Vocab): The target vocabulary.
        device (torch.device): The device to run the model on (e.g., "cpu" or "cuda").
        max_padding (int): The maximum padding length for sequences.
        pad_id (int): The padding token index, default is 2 which is <blank>.

    Returns:
        tuple: A tuple containing the source and target tensors, both padded to the maximum length.
    """
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab.lookup_indices(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab.lookup_indices(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            F.pad(
                processed_src,
                (0, max_padding - len(processed_src)),
                value=pad_id,
            )
        )
        tgt_list.append(
            F.pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)
