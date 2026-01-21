import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

import src.preprocessing as preprocessing
import src.nn_models as nn_models
import src.chess_trainer as chess_trainer


def export_onnx(model, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Dummy input with correct shape
    # important to use batch > 1, otherwise 1 batch-evaluation doesn't work
    dummy = torch.zeros(4, *preprocessing.BOARD_SHAPE, dtype=torch.float32)

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


def main(nn_architecture_selector, dataset_selector, epochs):

    # load data
    df = pd.read_csv(f"data/cleaned/chess_train_val_{dataset_selector}.csv", nrows=None)
    fens = df["canonical_fen"]

    # create X and y
    y = df["advantage"].to_numpy(dtype=np.float32)
    path_X = Path(f"scratch/X_{dataset_selector}.dat")
    X = preprocessing.build_memmap(fens, path_X, np.float32)

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
    y = preprocessing.arcsinh(y)

    # training
    model_params = chess_trainer.train(
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
    architecture_choices = ["small", "medium", "large"]
    dataset_choices = ["small", "full"]

    default_architecture = architecture_choices[0]
    default_dataset = dataset_choices[0]
    default_epochs = 1

    parser = argparse.ArgumentParser(
        description="Train chess advantage model and export to ONNX."
    )
    parser.add_argument(
        "architecture",
        nargs="?",
        default=default_architecture,
        choices=architecture_choices,
        help=f"NN architecture to use (default: {default_architecture}).",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=default_dataset,
        choices=dataset_choices,
        help=f"Dataset to use (default: {default_dataset}).",
    )
    parser.add_argument(
        "epochs",
        nargs="?",
        type=int,
        default=default_epochs,
        help=f"Number of training epochs (default: {default_epochs}).",
    )

    args = parser.parse_args()

    nn_architecture_selector = args.architecture
    dataset_selector = args.dataset
    epochs = args.epochs

    print(f"Architecture: {nn_architecture_selector}")
    print(f"Dataset: {dataset_selector}")
    print(f"Epochs: {epochs}")

    main(nn_architecture_selector, dataset_selector, epochs)
