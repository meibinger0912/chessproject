import chess

def get_flipped_coordinates(row, col, flip):
    """Trả về tọa độ đã đảo nếu cần."""
    if flip:
        return 7 - row, 7 - col
    return row, col

def is_valid_move(board, move):
    """Kiểm tra xem nước đi có hợp lệ không."""
    return move in board.legal_moves

def get_legal_moves(board, selected_square):
    """Lấy danh sách các nước đi hợp lệ cho quân cờ được chọn."""
    if selected_square is None:
        return []
    return [
        move for move in board.legal_moves
        if move.from_square == selected_square
    ]