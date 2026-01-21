import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path


CHANNELS = {
    # us
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    # opponent pieces
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
    # castling rights (relative)
    "castling_us": 12,
    "castling_them": 13,
    # en passant target square
    "en_passant": 14,
}

NUM_CHANNELS = max(CHANNELS.values()) + 1
BOARD_SHAPE = (NUM_CHANNELS, 8, 8)


def board_to_array(fen: str):
    """produces board array from current players perspective"""
    board_fen, turn, castling, ep_square, *_ = fen.split(" ")
    rows = board_fen.split("/")

    arr = np.zeros(BOARD_SHAPE, dtype=np.float32)

    # Side to move
    us_is_white = turn == "w"

    def piece_channel(ch):
        """Map a FEN piece character to a relative channel index."""
        is_white_piece = ch.isupper()

        if is_white_piece == us_is_white:
            # our piece
            return CHANNELS[ch.upper()]
        else:
            # opponent's piece
            return CHANNELS[ch.lower()]

    # pieces
    for rank, row in enumerate(reversed(rows)):
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                ch_idx = piece_channel(ch)
                arr[ch_idx, rank, file] = 1.0
                file += 1

    # castling rights
    if castling != "-":
        if us_is_white:
            us_k, us_q = "K", "Q"
            them_k, them_q = "k", "q"
            us_rank, them_rank = 0, 7
        else:
            us_k, us_q = "k", "q"
            them_k, them_q = "K", "Q"
            us_rank, them_rank = 7, 0

        if us_k in castling:
            arr[CHANNELS["castling_us"], us_rank, 7] = 1.0
        if us_q in castling:
            arr[CHANNELS["castling_us"], us_rank, 0] = 1.0
        if them_k in castling:
            arr[CHANNELS["castling_them"], them_rank, 7] = 1.0
        if them_q in castling:
            arr[CHANNELS["castling_them"], them_rank, 0] = 1.0

    # En passant
    if ep_square != "-":
        file = ord(ep_square[0]) - ord("a")
        rank = int(ep_square[1]) - 1
        arr[CHANNELS["en_passant"], rank, file] = 1.0

    # swap sides, so that us always looks up the board
    if not us_is_white:
        arr = arr[:, ::-1, :]

    return arr


def build_memmap(fens, mmap_path: Path, dtype, override=False):
    n = len(fens)
    shape = (n, *BOARD_SHAPE)

    if (not override) and mmap_path.exists():
        mmap = np.memmap(mmap_path, dtype=dtype, mode="r", shape=shape)
    else:
        mmap = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=shape)
        for i, board in enumerate(tqdm(fens, desc="building memmap")):
            mmap[i] = board_to_array(board)
        mmap.flush()

    return mmap


def arcsinh(y):
    return np.arcsinh(y / 100.0)
