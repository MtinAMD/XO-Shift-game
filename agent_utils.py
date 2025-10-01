from typing import List, Optional, Tuple

EMPTY_CELL = None


def get_possible_selections(board: List[List[Optional[str]]], player_symbol: str) -> List[Tuple[int, int]]:
    """
    Finds all valid source cells a player can select from the rim.

    Args:
        board: The current game board state.
        player_symbol: The symbol of the player ('X' or 'O').

    Returns:
        A list of (row, col) tuples representing valid cells to select.
    """
    size = len(board)
    rim_cells_coords = []
    for r_idx in range(size):
        for c_idx in range(size):
            if r_idx == 0 or r_idx == size - 1 or c_idx == 0 or c_idx == size - 1:
                rim_cells_coords.append((r_idx, c_idx))

    empty_rim_selections = []
    player_rim_selections = []

    for r, c in rim_cells_coords:
        if board[r][c] == EMPTY_CELL:
            empty_rim_selections.append((r, c))
        elif board[r][c] == player_symbol:
            player_rim_selections.append((r, c))

    # The rule is: if empty cells exist on the rim, player MUST select one of them.
    # Otherwise, they must select one of their own pieces on the rim.
    if empty_rim_selections:
        return empty_rim_selections
    else:
        return player_rim_selections


def get_all_valid_moves(board: List[List[Optional[str]]], player_symbol: str) -> List[Tuple[int, int, int, int]]:
    """
    Generates a list of all possible valid moves for a given player and board state.

    Args:
        board: The current game board state.
        player_symbol: The symbol of the player making the move.

    Returns:
        A list of valid moves, where each move is a tuple (src_r, src_c, tgt_r, tgt_c).
    """
    size = len(board)
    possible_selections = get_possible_selections(board, player_symbol)

    if not possible_selections:
        return []

    all_genuinely_valid_moves: List[Tuple[int, int, int, int]] = []

    for sr, sc in possible_selections:
        candidate_targets_for_src = [
            (sr, 0),
            (sr, size - 1),
            (0, sc),
            (size - 1, sc)
        ]

        actual_targets_for_src = set()

        for tr, tc in candidate_targets_for_src:
            if sr == tr and sc == tc:  # Target cannot be the source itself
                continue
            actual_targets_for_src.add((tr, tc))

        for tr_final, tc_final in list(actual_targets_for_src):
            all_genuinely_valid_moves.append((sr, sc, tr_final, tc_final))

    return all_genuinely_valid_moves
