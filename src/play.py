#!/usr/bin/env python3
"""
play_chess_agents.py

Interactive script to play against ONNX chess agents using src.preprocessing.board_to_array.

Dependencies:
    pip install python-chess onnxruntime numpy

Usage:
    python play_chess_agents.py
"""
import os
import onnxruntime as ort
import numpy as np
import random
import chess
import traceback

from src.compare import choose_best_move


# -------------------------
# Agent wrapper
# -------------------------
class ONNXAgent:
    def __init__(self, session: ort.InferenceSession):
        self.session = session

    def predict(self, input_data: np.ndarray):
        input_name = self.session.get_inputs()[0].name
        arr = input_data.astype(np.float32)
        out = self.session.run(None, {input_name: arr})
        return out[0]


def load_onnx_agent(path: str):
    if not path:
        return None
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found at {path}")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return ONNXAgent(sess)


# -------------------------
# Model selection helpers
# -------------------------
def list_onnx_models(models_dir="models"):
    models_dir = os.path.expanduser(models_dir)
    if not os.path.isdir(models_dir):
        return []
    files = [f for f in os.listdir(models_dir) if f.lower().endswith(".onnx")]
    files.sort()
    return [os.path.join(models_dir, f) for f in files]


def display_models(models):
    if not models:
        print(f"No models found.")

    print(f"\nAvailable models:")
    for i, path in enumerate(models, start=1):
        print(f"  [{i}] {os.path.basename(path)}")


def choose_model_from_list(models, side_name):
    if not models:
        return ""

    while True:
        choice = ask(
            f"Choose {side_name} model by number (Enter for human): ",
            default="",
        )
        if choice == "":
            return ""
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                return models[idx - 1]
        except ValueError:
            pass
        print("Invalid selection.")


# -------------------------
# Chess helpers
# -------------------------
def display_board(board: chess.Board):
    board_str = board.unicode(invert_color=True, borders=True, empty_square=" ")
    print(board_str)


def parse_move(user_move: str, board: chess.Board):
    user_move = user_move.strip()
    if not user_move:
        return None

    try:
        move = chess.Move.from_uci(user_move)
        if move in board.legal_moves:
            return move
    except Exception:
        pass

    try:
        move = board.parse_san(user_move)
        if move in board.legal_moves:
            return move
    except Exception:
        pass

    return None


def ask(prompt: str, default: str = ""):
    try:
        val = input(prompt)
    except EOFError:
        return default
    if val.strip() == "":
        return default
    return val.strip()


# -------------------------
# Interactive loop
# -------------------------
def interactive_loop(white_agent, black_agent, epsilon=0.0):
    board = chess.Board()

    print("\nCommands:")
    print("  <move>  : play a move (UCI e2e4 or SAN Nf3)")
    print("  ai      : engine plays for current side")
    print("  hint    : show engine best move")
    print("  undo    : undo last move")
    print("  reset   : reset the board")
    print("  show    : re-display the board")
    print("  q       : quit\n")

    while True:
        print()
        display_board(board)

        side = "White" if board.turn == chess.WHITE else "Black"
        agent = white_agent if board.turn == chess.WHITE else black_agent
        human = agent is None

        cmd = ask(f"{side} to move ({'human' if human else 'agent'}): ")

        if cmd.lower() in ("q", "quit", "exit"):
            return

        if cmd.lower() == "show":
            continue

        if cmd.lower() == "reset":
            board.reset()
            continue

        if cmd.lower() == "undo":
            if board.move_stack:
                board.pop()
            continue

        if cmd.lower() == "hint":
            if agent is None:
                print("No agent for this side.")
                continue
            move = choose_best_move(board, agent, epsilon)
            print(f"Hint: {move.uci()} ({board.san(move)})")
            continue

        if cmd == "" or cmd.lower() == "ai":
            if agent is None:
                print("No agent for this side.")
                continue
            try:
                move = choose_best_move(board, agent, epsilon)
                print(f"Engine plays {move.uci()}")
            except Exception as e:
                print("Engine error:", e)
                traceback.print_exc()
        else:
            move = parse_move(cmd, board)
            if move is None:
                print("Invalid move.")
                continue

        board.push(move)

        outcome = board.outcome()
        if outcome is not None:
            display_board(board)
            print("Game over:", outcome)
            return


# -------------------------
# Main
# -------------------------
def main():
    print("\nPlay chess against agents")
    print("-------------------------")

    models = list_onnx_models("models")

    display_models(models)

    wpath = choose_model_from_list(models, "White")
    bpath = choose_model_from_list(models, "Black")

    white_agent = load_onnx_agent(wpath) if wpath else None
    black_agent = load_onnx_agent(bpath) if bpath else None

    print("\nStarting game. Type 'q' to quit.\n")
    interactive_loop(white_agent, black_agent)


if __name__ == "__main__":
    main()
