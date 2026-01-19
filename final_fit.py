import pandas as pd
import numpy as np
from pathlib import Path

import preprocess
import torch.nn as nn
import nn_models
import train

import torch
import onnx
from pathlib import Path


def export_onnx(model, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Dummy input with correct shape
    # batch = 1, channels = NUM_CHANNELS
    dummy = torch.zeros(1, *preprocess.BOARD_SHAPE, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["board"],
        output_names=["advantage"],
        dynamic_axes={
            "board": {0: "batch"},
            "advantage": {0: "batch"},
        },
        opset_version=18,
    )

    print(f"Exported ONNX model to {out_path}")


def main():
    nn_architecture_selector = "medium"  # `small`, `medium`, `large`
    dataset_selector = "small"  # `full` or `small`
    epochs = 5

    print(f"Architecture: {nn_architecture_selector}")
    print(f"Dataset: {dataset_selector}")

    # create y
    df = pd.read_csv(f"data/cleaned/chess_train_val_{dataset_selector}.csv", nrows=None)
    y = df["advantage"].to_numpy(dtype=np.float32)

    # create or load X
    path_X = Path(f"scratch/X_{dataset_selector}.dat")
    if path_X.exists():
        X = np.memmap(
            path_X, dtype=np.float32, mode="r", shape=(len(y), *preprocess.BOARD_SHAPE)
        )
    else:
        fens = df["canonical_fen"]
        X = preprocess.build_memmap(fens, path_X, np.float32)

    # select architecture
    if nn_architecture_selector == "small":
        nn_architecture = nn_models.SmallCNNFast
    elif nn_architecture_selector == "medium":
        nn_architecture = nn_models.MediumResNet
    elif nn_architecture_selector == "large":
        nn_architecture = nn_models.LargeResNetSE
    else:
        raise ValueError(f"Unknown architecture: {nn_architecture_selector}")

    # transform target
    y = np.arcsinh(y / 100.0)

    # training
    model_params = train.train(
        X,
        y,
        model=nn_architecture(),
        lr=1e-4,
        epochs=epochs,
        load_workers=1,
        batch_size=4096,
        val_split=0.01,
        seed=0,
        loss_function=nn.SmoothL1Loss(),
    )[0]

    model = nn_architecture()
    model.load_state_dict(model_params)
    model.eval()

    export_onnx(
        model,
        out_path=f"models/chess_eval-{dataset_selector}_data-{nn_architecture_selector}_model-{epochs}_epochs.onnx",
    )


if __name__ == "__main__":
    main()
