import chess
import src.preprocessing as preprocessing
import torch
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch.distributions import Categorical
import random
import onnxruntime as ort


class PyTorchAgent:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.inference_mode()
    def predict(self, input_data):
        tensor = torch.from_numpy(input_data).to(dtype=torch.float32)
        output = self.model(tensor)
        return output.numpy()


class ONNXAgent:
    def __init__(self, model_path_or_session):
        self.session = model_path_or_session

    def predict(self, input_data):
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_data})
        return output[0]


def choose_best_move(board: chess.Board, agent, epsilon):
    # collect numpy-arrays for each candidate move
    arrays = []
    moves = list(board.legal_moves)

    # forced move
    if len(moves) == 1:
        return moves[0]

    # if epsilon is or <= 0.0
    if epsilon and epsilon > 0.0:
        return random.choice(moves)

    for move in moves:
        board.push(move)
        fen = board.fen()
        arr = preprocessing.board_to_array(fen)
        arrays.append(arr)
        board.pop()

    batch = np.array(arrays)

    # their advantages
    preds = agent.predict(batch)
    preds = preds.squeeze()

    # best move -> least advantage for them
    return moves[int(np.argmin(preds))]


def fight(agent1, agent2, epsilon, max_random_rounds):
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
            move = choose_best_move(board, agent1, epsilon)
        else:
            move = choose_best_move(board, agent2, epsilon)
        board.push(move)

        outcome = board.outcome()

    return outcome, board.fen()


def war(
    agent1,
    agent2,
    games_per_side=10,
    epsilon=0.1,
    max_random_rounds=20,
):
    outcome = {"model1": 0, "model2": 0, "draw": 0}
    for _ in range(games_per_side):
        b1 = fight(agent1, agent2, epsilon, max_random_rounds)
        w1 = b1[0].winner

        if w1 is None:
            outcome["draw"] += 1
        elif w1 == "True":
            outcome["model1"] += 1
        else:
            outcome["model2"] += 1
        b2 = fight(agent1, agent2, epsilon, max_random_rounds)

        w2 = b2[0].winner
        if w2 is None:
            outcome["draw"] += 1
        elif w2 == "True":
            outcome["model2"] += 1
        else:
            outcome["model1"] += 1
    return outcome
