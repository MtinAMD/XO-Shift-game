# XO-Shift Game ❌⭕️

A new version of traditional Tic-Tac-Toe (XO) game for Windows, built in Python.
- Play against a friend locally on your computer
- Play against two different AI agents
- Let the agents play with each other and see how they play

---

# Game Description

The game is played on a 3×3, 4×4, or 5×5 grid.
In this game, the ultimate goal, similar to Tic-Tac-Toe (XO), is to form a row, column, or diagonal with your own pieces. However, in this version of the game, the method of placing pieces on the board is slightly different.
In each turn, you must select two valid cells on the board and perform a shift operation based on the two selected cells.
To select the first cell: If there is an empty cell on the perimeter of the board, you are required to choose one of them. If there are no empty cells on the perimeter, you must choose from among the cells on the perimeter that are occupied by your own pieces.
To select the second cell: You must choose a cell that is at the beginning or end of the same row or column as the first cell you selected.
Note that the first and second selected cells cannot be the same.
Ultimately, the piece from the first cell moves into the position of the second cell, shifting the other cells along that path.
The number of moves that can occur in a game is limited. If the number of moves reaches 200 and no one has won, the match may be considered a draw.

---

## Prerequisites 🛠️

Make sure you have **Python 3.x** installed on your system
You can download here: [https://www.python.org/downloads/](https://www.python.org/downloads/)

then instal pygame :
```bash
pip install pygame
```

---

## Run 💾

Clone this repository:

```bash
git clone https://github.com/MtinAMD/XO-Shift-game.git
cd XO-Shift-game
run main.py
```

## Usage

You can play in 3 modes of 3×3, 4×4, or 5×5
the default agent is the hardest one.
if you want to play with an easier agent simply edit line 71 and 72 of main.py:
```bash
agent1_path_config = "hard_agent.py"
agent2_path_config = "random_agent.py"
```
