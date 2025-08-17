import argparse
import yaml
import torch
import sklearn
from engine import engine
from model import model_builder
from data_setup import data_setup
from data_setup import load_vocab
from data_setup import load_tokenizers


def main():
    """Main function to run the training."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Extracting config file...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config_file_name = args.config
    try:
        with open(config_file_name, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Config file not found. Exiting.")
        return

    print("Loading tokenizers...")
    spacy_ja, spacy_en = load_tokenizers.load_tokenizers()

    print("Extracting data...")
    raw_df = data_setup.extract_data(config["root_path"])
    if raw_df.empty:
        print("No data found. Exiting.")
        return
    train_df, test_df = sklearn.model_selection.train_test_split(
        raw_df, test_size=config["train_test_split"]
    )

    if not config["train_data_limit"] == -1:
        num = config["train_data_limit"]
        train_df = train_df[:num]

    if not config["test_data_limit"] == -1:
        num = config["test_data_limit"]
        test_df = test_df[:num]

    print("Creating vocabulary...")
    vocab_src, vocab_tgt = load_vocab.load_vocab(
        spacy_ja, spacy_en, raw_df, config["vocab_path"]
    )

    print("Creating dataloaders...")
    train_dataloader = data_setup.create_dataloaders(
        train_df,
        device=device,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_ja=spacy_ja,
        spacy_en=spacy_en,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
    )
    test_dataloader = data_setup.create_dataloaders(
        test_df,
        device=device,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_ja=spacy_ja,
        spacy_en=spacy_en,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
    )
    if train_dataloader is None or test_dataloader is None:
        print("No data found in DataLoader. Exiting.")
        return

    print("Building model...")
    d_model = 512
    model = model_builder.build_model(
        src_vocab=len(vocab_src),
        tgt_vocab=len(vocab_tgt),
        N=config["num_layers"],
        d_model=d_model,
    )

    model.to(device)

    print("Start training...")
    engine.train_worker(
        model=model,
        d_model=d_model,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        config=config,
        device=device,
    )


if __name__ == "__main__":
    main()
