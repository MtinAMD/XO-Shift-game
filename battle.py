#!/usr/bin/env python3
import argparse
import importlib
import importlib.util
import multiprocessing as mp
import queue
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Any

# Import XOShift core from project (assumed in PYTHONPATH)
try:
    from game import XOShiftGame
except Exception as e:
    print("Error: Could not import 'game.XOShiftGame'. Make sure you run this next to the project files.", file=sys.stderr)
    raise

# ---- Agent loading helpers ----

def _import_module_by_path(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from path: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_agent(alias: str) -> Callable:
    """Try to import an agent by alias:
    - If alias looks like a filepath ending with .py -> import by path
    - Else try 'agents.<alias>.agent_move'
    - Else try '<alias>.agent_move' (module in cwd)
    - Else try '<alias>.py' in cwd by path
    Returns the callable agent_move(board, player_symbol).
    """
    if alias.endswith('.py'):
        mod = _import_module_by_path(alias, f"agent_mod_{alias.replace('/', '_').replace('\\', '_')}")
        if not hasattr(mod, 'agent_move'):
            raise AttributeError(f"Agent file '{alias}' has no 'agent_move' function.")
        return getattr(mod, 'agent_move')

    # try agents.<alias>
    try:
        mod = importlib.import_module(f"agents.{alias}")
        if hasattr(mod, 'agent_move'):
            return getattr(mod, 'agent_move')
    except ModuleNotFoundError:
        pass

    # try <alias> as module in cwd
    try:
        mod = importlib.import_module(alias)
        if hasattr(mod, 'agent_move'):
            return getattr(mod, 'agent_move')
    except ModuleNotFoundError:
        pass

    # try <alias>.py by path in cwd
    try:
        mod = _import_module_by_path(f"{alias}.py", f"agent_mod_{alias}")
        if hasattr(mod, 'agent_move'):
            return getattr(mod, 'agent_move')
    except Exception:
        pass

    raise ImportError(f"Could not load agent '{alias}'. Try giving a path like 'agents/your_agent.py'.")


# ---- Match engine (headless) ----

AGENT_TIME_LIMIT = 2.0
MAX_TURNS = 250

def _agent_process_wrapper(agent_fn: Callable, board_copy: List[List[Optional[str]]], player_symbol: str, q: mp.Queue):
    try:
        move = agent_fn(board_copy, player_symbol)
        q.put(move)
    except Exception as e:
        q.put(e)

@dataclass
class GameResult:
    winner: str  # 'X', 'O', or 'Draw'
    turns: int
    timeouts_X: int = 0
    timeouts_O: int = 0
    invalids_X: int = 0
    invalids_O: int = 0

def play_one(agent_X: Callable, agent_O: Callable, board_size: int, time_limit: float) -> GameResult:
    game = XOShiftGame(size=board_size)
    turn_count = 0
    res = GameResult(winner='Draw', turns=0)

    while not game.winner and turn_count < MAX_TURNS:
        player = game.current_player  # 'X' or 'O'
        agent_fn = agent_X if player == 'X' else agent_O

        # copy board for the agent process
        board_copy = [[cell for cell in row] for row in game.board]
        q: mp.Queue = mp.Queue()
        p = mp.Process(target=_agent_process_wrapper, args=(agent_fn, board_copy, player, q))
        p.start()

        move, err, timed_out = None, None, False
        try:
            out = q.get(timeout=time_limit)
            if isinstance(out, Exception):
                err = out
            else:
                move = out
        except queue.Empty:
            timed_out = True
        except Exception as e:
            err = e

        if p.is_alive():
            p.terminate()
        p.join(timeout=0.5)
        if p.is_alive():
            p.kill()
            p.join()

        if timed_out:
            if player == 'X':
                res.timeouts_X += 1
            else:
                res.timeouts_O += 1
            # timeout counts as a turn (consistent with main.py), and opponent gets the turn
            turn_count += 1
            game.switch_player()
            continue
        if err is not None:
            # crash -> just switch player
            if player == 'X':
                res.invalids_X += 1
            else:
                res.invalids_O += 1
            game.switch_player()
            continue

        # Validate move shape
        if not (isinstance(move, (tuple, list)) and len(move) == 4 and all(isinstance(x, int) for x in move)):
            if player == 'X':
                res.invalids_X += 1
            else:
                res.invalids_O += 1
            game.switch_player()
            continue

        sr, sc, tr, tc = move
        if game.apply_move(sr, sc, tr, tc, player):
            turn_count += 1
            if not game.winner:
                game.switch_player()
        else:
            # invalid move
            if player == 'X':
                res.invalids_X += 1
            else:
                res.invalids_O += 1
            game.switch_player()

    if game.winner:
        res.winner = game.winner
    else:
        res.winner = 'Draw'
    res.turns = turn_count
    return res


# ---- Tournament ----

def run_tournament(agent_a_alias: str, agent_b_alias: str, games: int, board_size: int, time_limit: float) -> None:
    agent_a = load_agent(agent_a_alias)
    agent_b = load_agent(agent_b_alias)

    wins_a = wins_b = draws = 0
    timeouts_a = timeouts_b = 0
    invalids_a = invalids_b = 0
    total_turns = 0

    # Alternate colors (A is X on even, O on odd)
    for g in range(games):
        if g % 2 == 0:
            res = play_one(agent_a, agent_b, board_size, time_limit)
            # A played X
            if res.winner == 'X':
                wins_a += 1
            elif res.winner == 'O':
                wins_b += 1
            else:
                draws += 1
            timeouts_a += res.timeouts_X
            timeouts_b += res.timeouts_O
            invalids_a += res.invalids_X
            invalids_b += res.invalids_O
        else:
            res = play_one(agent_b, agent_a, board_size, time_limit)
            # A played O
            if res.winner == 'X':
                wins_b += 1
            elif res.winner == 'O':
                wins_a += 1
            else:
                draws += 1
            timeouts_b += res.timeouts_X
            timeouts_a += res.timeouts_O
            invalids_b += res.invalids_X
            invalids_a += res.invalids_O

        total_turns += res.turns

        if (g + 1) % max(1, games // 10) == 0:
            print(f"Progress: {g + 1}/{games} games...")

    print("\n=== Tournament Results ===")
    print(f"Games:           {games}")
    print(f"Board size:      {board_size}x{board_size}")
    print(f"Time/turn:       {time_limit:.2f}s\n")

    print(f"Agent A ({agent_a_alias}) wins: {wins_a}")
    print(f"Agent B ({agent_b_alias}) wins: {wins_b}")
    print(f"Draws:                           {draws}")
    print(f"Win rate A (excl. draws):        {wins_a / max(1, (wins_a + wins_b)) * 100:.1f}%\n")

    print(f"Timeouts A / B:  {timeouts_a} / {timeouts_b}")
    print(f"Invalids A / B:  {invalids_a} / {invalids_b}")
    print(f"Avg turns/game:  {total_turns / games:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Headless XOShift tournament runner.")
    parser.add_argument('--games', type=int, default=500, help='Number of games to play (default: 500)')
    parser.add_argument('--size', type=int, default=5, choices=[3,4,5], help='Board size (3/4/5). Default: 5')
    parser.add_argument('--time', type=float, default=2.0, help='Per-move time limit in seconds (default: 2.0)')
    parser.add_argument('--agent-a', type=str, default='your_agent', help="Agent A (module name or path). Default: your_agent")
    parser.add_argument('--agent-b', type=str, default='sample_agent', help="Agent B (module name or path). Default: sample_agent")
    args = parser.parse_args()

    # safer on Windows for multiprocessing
    mp.freeze_support()

    run_tournament(args.agent_a, args.agent_b, args.games, args.size, args.time)

if __name__ == '__main__':
    main()
