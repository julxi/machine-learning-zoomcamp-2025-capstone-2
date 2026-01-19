import chess
import preprocess
import torch
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch.distributions import Categorical
import random


@torch.inference_mode()
def choose_best_move(board: chess.Board, model: nn.Module, epsilon):
    model.eval()

    # collect numpy-arrays for each candidate move
    arrays = []
    moves = list(board.legal_moves)

    # forced move
    if len(moves) == 1:
        return moves[0]

    # if epsilon is or <= 0.0
    if not epsilon or epsilon <= 0.0:
        return random.choice(moves)

    for move in moves:
        board.push(move)
        fen = board.fen()
        arr = preprocess.board_to_array(fen)
        arrays.append(arr)
        board.pop()

    batch_np = np.stack(arrays, axis=0)

    batch = torch.from_numpy(batch_np).to(dtype=torch.float32)

    # their advantage
    preds = model(batch)
    preds = preds.squeeze()

    # best move -> least advantage for them
    return moves[int(torch.argmin(preds))]


def fight(model1: nn.Module, model2: nn.Module, epsilon, max_random_rounds):
    board = chess.Board()

    outcome = board.outcome()

    turn = 0
    while outcome is None:
        # stop random moves after max_random_rounds
        turn += 1
        if max_random_rounds and turn > max_random_rounds * 2:
            epsilon = None

        # models choose moves
        if board.turn == chess.WHITE:
            move = choose_best_move(board, model1, epsilon)
        else:
            move = choose_best_move(board, model2, epsilon)
        board.push(move)

        outcome = board.outcome()

    return outcome, board.fen()


def war(
    model1: nn.Module,
    model2: nn.Module,
    games_per_side=10,
    epsilon=0.1,
    max_random_rounds=20,
):
    outcome = {"model1": 0, "model2": 0, "draw": 0}
    for _ in range(games_per_side):
        b1 = fight(model1, model2, epsilon, max_random_rounds)
        w1 = b1[0].winner

        if w1 is None:
            outcome["draw"] += 1
        elif w1 == "True":
            outcome["model1"] += 1
        else:
            outcome["model2"] += 1
        b2 = fight(model2, model1, epsilon, max_random_rounds)

        w2 = b2[0].winner
        if w2 is None:
            outcome["draw"] += 1
        elif w2 == "True":
            outcome["model2"] += 1
        else:
            outcome["model1"] += 1
    return outcome


from tqdm import tqdm


def rank_model(models, games_per_pair=10, epsilon=0.1, max_random_rounds=20):
    n = len(models)

    # initialise stats
    stats = {
        i: {"index": i, "points": 0.0, "wins": 0, "draws": 0, "losses": 0, "games": 0}
        for i in range(n)
    }
    pairwise = {}

    # round-robin
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            res = war(
                models[i],
                models[j],
                games_per_side=games_per_pair,
                epsilon=epsilon,
            )

            wins_i = int(res.get("model1", 0))
            wins_j = int(res.get("model2", 0))
            draws = int(res.get("draw", 0))
            total = int(res.get("total_games", wins_i + wins_j + draws))

            # update pairwise store
            pairwise[(i, j)] = {
                "wins_i": wins_i,
                "wins_j": wins_j,
                "draws": draws,
                "total_games": total,
            }

            # convert to points
            points_i = wins_i * 1.0 + draws * 0.0 - wins_j * 1.0
            points_j = -points_i

            # update global stats
            stats[i]["points"] += points_i
            stats[j]["points"] += points_j

            stats[i]["wins"] += wins_i
            stats[j]["wins"] += wins_j
            stats[i]["draws"] += draws
            stats[j]["draws"] += draws

            stats[i]["losses"] += wins_j
            stats[j]["losses"] += wins_i

            stats[i]["games"] += total
            stats[j]["games"] += total

    ranking = sorted(
        [stats[k] for k in stats], key=lambda x: (-x["points"], -x["wins"], x["index"])
    )
    return {"ranking": ranking, "pairwise": pairwise, "summary": stats}
