from typing import List, Optional, Tuple
import time
import math
import random

from agent_utils import get_all_valid_moves

Move = Tuple[int, int, int, int]

def agent_move(board: List[List[Optional[str]]], player_symbol: str) -> Move:
    """
      - Win now & don't allow los next checks
      - Negamax + alpha-beta (depth 1..4 adaptively)
      - Cheap but strong heuristic (line potential + center contrrol)
    """
    start = time.perf_counter()

    # lower if CPU is slower
    THINK_BUDGET = 1.4

    def time_left() -> float:
        return THINK_BUDGET - (time.perf_counter() - start)

    n = len(board)
    me = player_symbol
    opp = 'O' if me == 'X' else 'X'

    # Generate root moves
    root_moves = get_all_valid_moves(board, me)
    if not root_moves:
        return (0, 0, 0, 0)

    # Precompute static helpers
    lines = _precompute_lines(n)
    center_w = _precompute_center_weights(n)
    base_eval = _evaluate(board, me, lines, center_w)

    # 1. WIN-NOW: if any move wins immediately, play it
    for m in root_moves:
        if time_left() <= 0.01:
            return root_moves[0]
        child = _simulate(board, m, me)
        if _won(child, me):
            return m

    # 2. DON'T-LOSE-NEXT: prefer moves that avoid giving opponent an immediate win
    # If current pos lets opponent win next, try to neutralize
    opp_wins_now = _count_immediate_wins(board, opp)
    safe_moves = []
    if opp_wins_now > 0:
        best_m, best_score = None, -10**18
        for m in root_moves:
            if time_left() <= 0.01:
                # If we run out of time, at least pick something good
                return best_m if best_m else root_moves[0]
            child = _simulate(board, m, me)
            child_opp_wins = _count_immediate_wins(child, opp)
            if child_opp_wins == 0:
                # fully safe move; choose best by evaluation
                s = _evaluate(child, me, lines, center_w)
                if s > best_score:
                    best_score, best_m = s, m
                safe_moves.append(m)
        if best_m:
            return best_m
        # If we couldn't fully neutralize, still prefer ones that reduce immediate losses
        best_m2, best_score2 = None, -10**18
        for m in root_moves:
            if time_left() <= 0.01:
                return best_m2 if best_m2 else root_moves[0]
            child = _simulate(board, m, me)
            child_opp_wins = _count_immediate_wins(child, opp)
            # fewer opponent wins_next is better
            s = -100000 * child_opp_wins + _evaluate(child, me, lines, center_w)
            if s > best_score2:
                best_score2, best_m2 = s, m
        return best_m2 if best_m2 else root_moves[0]

    # 3. Order root moves (cheaply): static delta + center bias + random jitter
    root_scored = []
    for m in root_moves:
        if time_left() <= 0.01:
            # If we’re out of time, just pick the best seen so far
            root_scored.sort(key=lambda x: x[0], reverse=True)
            return root_scored[0][1] if root_scored else root_moves[0]
        child = _simulate(board, m, me)
        s = _evaluate(child, me, lines, center_w) - base_eval
        # tiny jitter to avoid ties causing worst-case alpha-beta behavior
        s += random.randint(0, 3)
        root_scored.append((s, m))
    root_scored.sort(key=lambda x: x[0], reverse=True)
    ordered_root = [m for _, m in root_scored]

    # 4. Iterative deepening negamax with alpha-beta (depth 1..4 as time allows)
    best_move = ordered_root[0]
    best_val = -math.inf
    # Max depth is intentionally small; deeper gets risky under Windows spawn
    for depth in (1, 2, 3, 4):
        if time_left() <= 0.05:
            break
        val, mv, completed = _search_root(board, me, depth, ordered_root, lines, center_w, time_left)
        if completed and (mv is not None):
            best_val, best_move = val, mv
        else:
            break  # out of time inside this depth

    return best_move


# Negamax + alpha-beta (root & internal)

def _search_root(board, me, depth, moves, lines, center_w, time_left) -> Tuple[int, Optional[Move], bool]:
    opp = 'O' if me == 'X' else 'X'
    alpha, beta = -10**9, 10**9

    best_val = -math.inf
    best_move = None

    for i, m in enumerate(moves):
        if time_left() <= 0.0:
            return best_val, best_move, False
        child = _simulate(board, m, me)
        if _won(child, me):
            return 10**8 - i, m, True  # immediate win

        val = -_negamax(child, opp, me, depth - 1, -beta, -alpha, lines, center_w, time_left)

        if val > best_val:
            best_val = val
            best_move = m
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break

    # If nothing evaluated (like ran out of time at start) fall back
    return best_val if best_move else -10**7, best_move, True


def _negamax(board, to_move, root, depth, alpha, beta, lines, center_w, time_left) -> int:
    if time_left() <= 0.0:
        # When time is gone, return a stand-pat eval — *critical* to stay under 2s.
        return _evaluate(board, root, lines, center_w)

    me = to_move
    opp = 'O' if me == 'X' else 'X'

    # Terminal
    if _won(board, opp):
        # opponent just made a line on their last move -> losing for `me`
        return -10**8 + (5 - depth)

    if depth <= 0:
        return _evaluate(board, root, lines, center_w)

    moves = get_all_valid_moves(board, me)
    if not moves:
        return _evaluate(board, root, lines, center_w)

    # quick win move first
    ordered = []
    for m in moves:
        child = _simulate(board, m, me)
        if _won(child, me):
            # immediate winning reply
            return 10**8
        # cheap ordering: static eval only at frontier
        ordered.append((_evaluate(child, root, lines, center_w), m))
        if time_left() <= 0.0:
            break
    ordered.sort(key=lambda x: x[0], reverse=True)
    ordered_moves = [m for _, m in ordered]

    best = -math.inf
    a = alpha
    for m in ordered_moves:
        if time_left() <= 0.0:
            break
        child = _simulate(board, m, me)
        val = -_negamax(child, opp, root, depth - 1, -beta, -a, lines, center_w, time_left)
        if val > best:
            best = val
        if val > a:
            a = val
        if a >= beta:
            break
    return best


# Evaluation & helpers

def _evaluate(bd: List[List[Optional[str]]], root: str, lines, center_w) -> int:
    """Positive is good for `root`."""
    opp = 'O' if root == 'X' else 'X'
    n = len(bd)

    if _won(bd, root):
        return 10**8
    if _won(bd, opp):
        return -10**8

    # Line potential (unblocked lines only) + threat bonus for N-1
    W = _weights_for_n(n)
    score = 0

    for line in lines:
        c_root = 0
        c_opp = 0
        for (r, c) in line:
            v = bd[r][c]
            if v == root:
                c_root += 1
            elif v == opp:
                c_opp += 1
        if c_root and c_opp:
            continue  # blocked
        if c_opp == 0:
            score += W[c_root]
            if c_root == n - 1:
                score += 1200 * n
        if c_root == 0:
            score -= W[c_opp]
            if c_opp == n - 1:
                score -= 1200 * n

    # Center control (Chebyshev rings)
    for r in range(n):
        row = bd[r]
        cw_row = center_w[r]
        for c in range(n):
            v = row[c]
            if v == root:
                score += cw_row[c]
            elif v == opp:
                score -= cw_row[c]

    return score


def _weights_for_n(n: int) -> List[int]:
    # escalating but safe integer weights for clean lines with i stones
    w = [0] * (n + 1)
    for i in range(1, n + 1):
        w[i] = (i * i * (10 + n))  # quadratic growth scaled by board size
    w[n] = 10**8  # (handled by terminal check)
    return w


def _precompute_lines(n: int):
    lines = []
    for r in range(n):
        lines.append([(r, c) for c in range(n)])
    for c in range(n):
        lines.append([(r, c) for r in range(n)])
    lines.append([(i, i) for i in range(n)])
    lines.append([(i, n - 1 - i) for i in range(n)])
    return lines


def _precompute_center_weights(n: int):
    mid = (n - 1) / 2.0
    M = [[0] * n for _ in range(n)]
    base = 8  # center influence scale
    for r in range(n):
        for c in range(n):
            dist = max(abs(r - mid), abs(c - mid))
            M[r][c] = int((n - dist) * base)
    return M


def _won(bd: List[List[Optional[str]]], sym: str) -> bool:
    n = len(bd)
    # rows
    for r in range(n):
        row = bd[r]
        good = True
        for c in range(n):
            if row[c] != sym:
                good = False
                break
        if good:
            return True
    # cols
    for c in range(n):
        good = True
        for r in range(n):
            if bd[r][c] != sym:
                good = False
                break
        if good:
            return True
    # diags
    good = True
    for i in range(n):
        if bd[i][i] != sym:
            good = False
            break
    if good:
        return True
    good = True
    for i in range(n):
        if bd[i][n - 1 - i] != sym:
            good = False
            break
    return good


def _simulate(bd: List[List[Optional[str]]], move: Move, sym: str) -> List[List[Optional[str]]]:
    """
    functional simulate of XOShift (mirrors game.apply_move).
    """
    sr, sc, tr, tc = move
    n = len(bd)
    new = [row[:] for row in bd]

    if sr == tr:  # horizontal shift
        step = -1 if tc < sc else 1
        c = sc
        while c != tc:
            new[sr][c] = new[sr][c + step]
            c += step
    else:         # vertical shift
        step = -1 if tr < sr else 1
        r = sr
        while r != tr:
            new[r][sc] = new[r + step][sc]
            r += step
    new[tr][tc] = sym
    return new


def _count_immediate_wins(bd: List[List[Optional[str]]], sym: str) -> int:
    """How many legal moves let `sym` win immediately from bd."""
    wins = 0
    for m in get_all_valid_moves(bd, sym):
        child = _simulate(bd, m, sym)
        if _won(child, sym):
            wins += 1
    return wins