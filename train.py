import numpy as np
from pathlib import Path
from tqdm import tqdm
import copy

from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch
import torch.nn as nn
import math
import topN


class ChessDataset(IterableDataset):
    """
    SIMPLE single-producer IterableDataset that yields *batches* (x_t, y_t).
    This implementation does NOT shard across multiple DataLoader workers; it
    expects either num_workers == 0 (main process loads) or num_workers == 1
    (one worker subprocess loads). If you start DataLoader with num_workers > 1,
    this dataset will raise a helpful error to avoid accidental duplicate data.
    """

    def __init__(
        self,
        X,
        y: np.ndarray,
        indices,
        batch_size,
        dtype=np.float32,
    ):
        super().__init__()
        self.X = X
        self.y = np.asarray(y)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.total_len = len(self.y)
        self.dtype = dtype

    def __len__(self):
        # number of batches (useful for progress bars)
        return int(math.ceil(len(self.indices) / self.batch_size))

    @property
    def num_samples(self):
        # total number of samples represented by this dataset (useful for averaging loss)
        return int(len(self.indices))

    def __iter__(self):
        # Explicitly disallow num_workers > 1 to keep logic simple and avoid accidental duplication.
        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError(
                "ChessDataset is single-producer only. "
                "Please set DataLoader(num_workers=0) or DataLoader(num_workers=1)."
            )

        if len(self.indices) == 0:
            return

        for i in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[i : i + self.batch_size]

            x_np = self.X[batch_idx]
            y_np = self.y[batch_idx]

            # random left-right mirror with p=0.5
            if np.random.random() < 0.5:
                # flip files: (C, 8, 8) -> mirror along last dimension
                x_np = np.flip(x_np, axis=-1).copy()

            x_t = torch.from_numpy(x_np)
            y_t = torch.from_numpy(y_np)

            yield x_t, y_t


def train_loop(model, opt, loader, dataset, loss_function):
    model.train()
    total_loss = 0.0
    processed = 0

    pbar = tqdm(
        loader, total=len(dataset), desc="train", unit="batches", smoothing=0.05
    )
    for xb, yb in pbar:
        opt.zero_grad()
        preds = model(xb)
        loss = loss_function(preds, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        processed += bs
        pbar.set_postfix({"avg_loss": total_loss / processed})
    pbar.close()
    return total_loss


@torch.no_grad()
def valid_loop(model, loader, dataset, loss_function):
    model.eval()
    total_loss = 0.0
    processed = 0

    pbar = tqdm(
        loader, total=len(dataset), desc="valid", unit="batches", smoothing=0.05
    )
    for xb, yb in pbar:
        preds = model(xb)
        loss = loss_function(preds, yb)
        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        processed += bs
        pbar.set_postfix({"avg_loss": total_loss / processed})
    pbar.close()
    return total_loss


def train(
    X,
    y,
    model,
    lr=1e-4,
    epochs=3,
    load_workers=1,
    batch_size=4096,
    val_split=0.1,
    seed=0,
    loss_function=None,
    keep_best=1,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # fail early
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    # I'm not sure if this is necessary, but I don't want to risk it
    if X.dtype != np.float32:
        raise ValueError("X must be float32")
    if y.dtype != np.float32:
        raise ValueError("y must be float32")

    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_n = int(n * val_split)
    train_idx = idx[:-val_n]
    val_idx = idx[-val_n:]

    train_ds = ChessDataset(X, y, train_idx, batch_size=batch_size)
    val_ds = ChessDataset(X, y, val_idx, batch_size=batch_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=load_workers,
        persistent_workers=(load_workers > 0),
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=load_workers,
        persistent_workers=(load_workers > 0),
    )

    if loss_function is None:
        loss_function = nn.functional.mse_loss
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_models = topN.TopN(keep_best)
    model.compile()
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs} — lr={opt.param_groups[0]["lr"]:.5g}')

        param_snap = [p.detach().clone() for p in list(model.parameters())]

        train_loss_sum = train_loop(model, opt, train_loader, train_ds, loss_function)
        val_loss_sum = valid_loop(model, val_loader, val_ds, loss_function)

        train_loss = train_loss_sum / train_ds.num_samples
        val_loss = val_loss_sum / val_ds.num_samples

        best_models.push(val_loss, copy.deepcopy(model.state_dict()))
        print(f"  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        scheduler.step()

        param_change_l1 = 0.0
        for orig, p in zip(param_snap, list(model.parameters())):
            param_change_l1 += float((p.detach().cpu() - orig).abs().sum())
        print(f"PARAM CHANGE (average): {param_change_l1/len(param_snap):.6g}")

    return best_models.get_elements()
