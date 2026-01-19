# Setting

I like ML a bit more than supervised learning. So I thought maybe I could train a chess engine with some a-class training material (I'm not a chess player btw).

I had great plans for this, like being able to play against the derpy agents, but sadly I've run out of time, to finish all of it. So I had to stop at the deploy stage level.

Also I couldn't do proper fine-tuning or experiments with different neural network architectures since training just takes too long and also comparing losses for different loss functions or target transformations is tricky.

**Heads-up**: There is no docker nor cloud deployment 🚫.

# 1. Problem Description

I've got a dataset from Kaggle https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations. The dataset comes from this project https://github.com/r2dev2/ChessDataContributor/tree/master that collectively used stockfish to analyse chess positions.

It has only two fields

- fen (string notation for chess positions)
- evaluation

What we basically want is to train a model that gets a fen and turns that into an evaluation.

The ideal workflow is as follows:
```
Exploratory Data Analysis (EDA) + Feature Engineering
→ Try out Neural Network with different loss functions, target transformation, and architecture
→ Train Script with Best Parameters and onnxing the model
→ Creat a Deployment Script
```

# 2. Data Description

The column 'fen' contains the [Forsyth–Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) of a chess position. It describes a chess position. An example looks like this `R1k5/8/1pK5/1r1P3p/P4r2/8/8/7r b - - 0 50
`. The string consists of 6 parts separated by spaces. The first part describes the board, which in our example looks like this:

![example_fen](/pictures/example_fen.svg)

The second part of the string encodes the colours who's turn it is. In our case it's `b` for black.

The third and fourth part encode en-passant squares and castling rights. Here they are `-` which means no en-passant square and no castling rights.

The final two numbers are clocks. The last one is the total clock, ticking up after both player have moved and the other number is a clock for pawn moves. For the details I refer to the wikipedia article. What is important about the clocks is, that they restrict the length of games, so even when two derpy neural network play against each other, their match will eventually finish in a draw.

The other column `evaluation` shows a stockfish evaluation of the position from whites perspective. It is a string but semantically it is distinguished into an advantage and mate counter.

The evaluation is a mate counter when it starts with a `#`. Then the next character is either a `+` or `-` which indicates if it's a mate for white or for black. The number following the sign indicates how many moves it will take to mate. For example `#-3` means that black can mate in 3 moves. If the number is `0` then the position describes a mate position, like above in the picture the evaluation is `#+0` as the fen describes a position in which black is mated.

If the evalutaion does not start with a `#`, it is the advantage of white. The more positive the more is white winning. The more negative the more is black winning. Advantage of `0` means the position is even (most likely a draw). The advantage is actually measured in centipawns, but that doesn't matter here.

**Note**: if you are wondering where this silly chess position of the example above comes from. It's from a game between two derpy chess engines.


# 3. EDA Summary

EDA and data preparation was done in the notebook `notebooks/01_data_preparation.ipynb`.
A big part of the EDA was finding out the conventions desribed in the previous section, though not present int the notebook anymore.

Mostly the notebook is about data preparation. The two main parts are:
- change evaluation from _white's perspective_ to _current player's perspective_ (more in the next section)
- turning `evaluation` into an integer `advantage` that takes extreme values if evaluation is a mate in X.
- identify and remove identical positions under mirroring of the position


A mirroring a position means taking a position like this one (sorry for the messy position)

![example_fen](/pictures/position_before_mirroring.svg)

and basically swapping the roles

![example_fen](/pictures/position_after_mirroring.svg)

the white player is now the black player and vice versa but both keep their positions and turns. A good evaluation should be invariant under mirroring. If white has `+100` centipawns in a position that black should have `+100` centipawns in the mirrored position.

There are also other things done in the notebook, but these are the main ones.

# 4. Modelling Approach & Metrics

The neural network models operate of tensors of shape `(15, 8, 8)`. The first channel encodes the piece (6 for white + 6 for black) and then there are 3 more channels for castling rules and en passant. The other two channels desrcibe the file and rank of the square. For example `board_array(0,1,1)` means that there is a white pawn on the square `b2` (actually I don't know if the exact position is correct which depends on the orientation, but as long as we use the same encoding function for training and deployment it's fine).

I don't know which neural network architecture is best for chess. So I just asked a LLM agent to give me three options (small, medium, big) and see which ones best.

The metric for determining which model is best was kind of the most fun and biggest let down for me at the same time. So instead of using MSE, MAE or any other metric for deciding which model is better, I just let them play against each other. Even though they can only evaluate a position we can use the evaluation to to look ahead and decide on the value of a move:
E.g. in the start position

![example_fen](/pictures/start_position.svg)

We can just emulate all possible starting moves, get the advantage (by design this is the advantage of the black player) for each of these resulting positions and choose the move that results in the smallest advantage for black.

So in principle a great idea but in practice it didn't really work. All models that I've trained did perform basically equally well. Why I think this is, I'll explain in the last section about limitations.


# 5. How to run and what

I managed this project using `uv`. For this you should have a global installation of `uv` and then run `uv sync` to install the dependencies.

If you use a differnet package manager you have to install the packages from `pyproject.toml`.

If anything doesn't work properly. I already apologize but I didn't have any time to clean up the repo (cluttered with python files) nor test the setup from scratch.

## Running EDA and Data Preparation

First download the `chess-evaluation.zip` from kaggle and unzip it into the `data` folder (you need an account for that, the data is too big to ship it in the repo, sorry).
Check that you have `data/chessData.csv` now. (We don't need the other files in the zip)

Now you can just open `01_data_preparation.ipynb` and run the cells (however you open jupyter notebooks, make sure that you use the venv that you created with `uv` for the notebook)


## Tuning the Training

For this you can just open `02_model_training.ipynb`. It uses the file `data/cleaned/chess_train_val_small.csv` which is included in the repo (and was generated by the previous notebook).

Note that the training cells need quite long to execute. And the results themselves are not very expressive.

## Run the final Training

You have to run `final_fit.py` for this (in uv).

One way (on Linux is)

Activate 
```bash
source .venv/bin/activate
python final_fit.py
```
This generates onnx files in `models`. You can change in the `main` which model architecture, how many epochs, and if you want to use the small training data or the full data. (The full data is only available if you ran the date preparation notebook)

## Running the Server Locally
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

and you cen test it with these commands

#### Health
```bash
curl -X GET "http://localhost:8000/health"
```

#### Predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'

```


## 7. Known limitations / next steps

After nearly two hours: medium-loss: 0.371 for mini: 0.56 and plateauing.

-TBD