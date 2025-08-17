import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from engine import engine_utils
import data_setup.data_utils as data_utils


class TrainState:
    """A class to keep track of the training state."""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter: iter,
    model: nn.Module,
    loss_compute: callable,
    optimizer: nn.Module,
    scheduler: nn.Module,
    mode="train",
    accum_iter: int = 1,
    train_state=TrainState(),
    device: torch.device = "cpu",  # Device to run the model on
):
    """Runs a single epoch of training or evaluation.

    Args:
        data_iter (iterable): An iterable that provides batches of data.
        model (nn.Module): The model to train or evaluate.
        loss_compute (callable): A function to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        mode (str): The mode of operation, either "train", "eval", or "train+log".
        accum_iter (int): Number of gradient accumulation steps.
        train_state (TrainState): An instance of TrainState to keep track of training state.

    Returns:
        tuple: A tuple containing the average loss and the updated train state.
    """

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        batch.src, batch.tgt, batch.src_mask, batch.tgt_mask = (
            batch.src.to(device),
            batch.tgt.to(device),
            batch.src_mask.to(device),
            batch.tgt_mask.to(device),
        )

        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens

            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1

            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start

            print(
                (
                    "Epoch Step: %6d | accumulation step: %3d | loss: %6.2f"
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def train_worker(
    model: nn.Module,
    d_model: int,
    vocab_src: data_utils.Vocab,
    vocab_tgt: data_utils.Vocab,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    config: dict,
    device: torch.device,
):
    """Function to train the model using the provided configuration and data.

    Args:
        model (nn.Module): The model to be trained.
        d_model (int): The dimension of the model.
        vocab_src (data_utils.Vocab): The source vocabulary.
        vocab_tgt (data_utils.Vocab): The target vocabulary.
        spacy_ja: The Japanese tokenizer.
        spacy_en: The English tokenizer.
        config (dict): Configuration dictionary containing training parameters.
    """
    print(f"Train worker process using device: {device} for training", flush=True)

    pad_idx = vocab_tgt["<blank>"]
    module = model

    criterion = engine_utils.LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: engine_utils.rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        model.train()

        print(f"Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (
                engine_utils.Batch(b_src, b_tgt, pad_idx)
                for b_src, b_tgt in train_dataloader
            ),
            model,
            engine_utils.SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
        torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"Epoch {epoch} Test ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (
                engine_utils.Batch(b_src.to(device), b_tgt.to(device), pad_idx)
                for b_src, b_tgt in test_dataloader
            ),
            model,
            engine_utils.SimpleLossCompute(module.generator, criterion),
            engine_utils.DummyOptimizer(),
            engine_utils.DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    file_path = f"{config['file_prefix']}final.pt"
    torch.save(module.state_dict(), file_path)
