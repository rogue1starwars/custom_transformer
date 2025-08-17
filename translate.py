import os
import yaml

import torch
from torch import nn
import torch.nn.functional as F

import model.model_builder as model_builder
import data_setup.data_setup as data_setup
import data_setup.load_vocab as load_vocab
import data_setup.load_tokenizers as load_tokenizers
import data_setup.data_utils as data_utils
import utils.helper_functions as helper_functions


def tokenize(text, spacy_tokenizer):
    """Tokenize a single text using the provided tokenizer."""
    return [tok.text for tok in spacy_tokenizer.tokenizer(text)]


def translate(
    model: nn.Module,
    sentence: str,
    spacy_ja: object,
    vocab_src: data_utils.Vocab,
    vocab_tgt: data_utils.Vocab,
    max_len: int,
    device: torch.device,
    pad_id: int,
):
    """Translate a single sentence from Japanese to English.

    Args:
        model (nn.Module): The translation model.
        sentence (str): The Japanese sentence to translate.
        spacy_ja (object): The Japanese tokenizer.
        vocab_src (data_utils.Vocab): The source vocabulary.
        vocab_tgt (data_utils.Vocab): The target vocabulary.
        max_len (int): The maximum length of the input sequence.
        device (torch.device): The device to run the model on.
        pad_id (int): The padding token ID.

    Returns:
        str: The translated English sentence.
    """
    model.eval()
    model.to(device)
    tokenize_ja = lambda text: tokenize(text, spacy_ja)
    tokens = tokenize_ja(sentence)
    indicies = vocab_src.lookup_indices(tokens)
    src = torch.tensor([indicies], dtype=torch.long, device=device)

    # Add start and end tokens and pad
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id

    processed_src = torch.cat(
        [
            bs_id,
            src.squeeze(0),
            eos_id,
        ],
        0,
    )

    src_tensors = F.pad(
        processed_src,
        (0, max_len - len(processed_src)),
        value=pad_id,
    ).unsqueeze(
        0
    )  # Add batch dimension back
    src_tensors.to(device)

    src_mask = (src_tensors != pad_id).unsqueeze(-2)
    src_mask.to(device)

    output_tensors = helper_functions.greedy_decode(
        model, src_tensors, src_mask, max_len, start_symbol=0
    )
    output = [vocab_tgt.itos[idx] for idx in output_tensors.squeeze(0).tolist()]

    # Remove start, end, and padding tokens from the output
    print(output_tensors)
    output_sentence = []
    for token in output:
        if token == "<s>":
            continue
        if token == "</s>":
            break
        if token == "<blank>":
            break
        output_sentence.append(token)

    return " ".join(output_sentence)


def main():
    """Main function to run the translation."""
    sentence = "こんにちは。今日は良い天気ですね。"
    max_padding = 72  # Use the same max_padding as in your training config

    print("Extracting config file...")
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Config file not found. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizers...")
    spacy_ja, spacy_en = load_tokenizers.load_tokenizers()

    print("Extracting data...")
    raw_df = data_setup.extract_data(config["root_path"])
    if raw_df.empty:
        print("No data found. Exiting.")
        return

    print("Loading vocabulary...")
    vocab_src, vocab_tgt = load_vocab.load_vocab(
        spacy_ja, spacy_en, raw_df, config["vocab_path"]
    )
    pad_id = vocab_src.get_stoi()["<blank>"]

    print("Loading model...")
    model_path = "./pretrained_models/jesec_model_final.pt"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist. Exiting.")
        return
    model = model_builder.build_model(
        src_vocab=len(vocab_src),
        tgt_vocab=len(vocab_tgt),
        N=config["num_layers"],
        d_model=512,  # Use the same d_model as in your training config
    )
    model.load_state_dict(torch.load(model_path, map_location=device))

    translated_sentence = translate(
        model=model,
        sentence=sentence,
        spacy_ja=spacy_ja,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        max_len=max_padding,
        device=device,
        pad_id=pad_id,
    )
    print(f"Japanese: {sentence}")
    print(f"English: {translated_sentence}")


if __name__ == "__main__":
    main()
