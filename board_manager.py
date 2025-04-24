import chess
from board_utils import get_flipped_coordinates, is_valid_move, get_legal_moves

class BoardManager:
    def __init__(self, player_side):
        self.board = chess.Board()
        self.selected_square = None
        self.player_side = player_side

    def handle_click(self, row, col):
        """Xử lý khi người chơi nhấp vào bàn cờ."""
        if self.player_side == "black":
            row, col = get_flipped_coordinates(row, col, flip=True)

        square = chess.square(col, row)
        if self.board.is_checkmate() or self.board.is_stalemate():
            return

        player_color = chess.WHITE if self.player_side == "white" else chess.BLACK

        if self.board.turn == player_color:
            if self.selected_square is None:
                if self.board.piece_at(square) and self.board.piece_at(square).color == player_color:
                    self.selected_square = square
            else:
                move = chess.Move(self.selected_square, square)
                if is_valid_move(self.board, move):
                    self.board.push(move)
                self.selected_square = None

    def get_legal_moves_for_selected_piece(self):
        """Lấy danh sách các nước đi hợp lệ cho quân cờ được chọn."""
        return get_legal_moves(self.board, self.selected_square)

    def ai_move(self, stockfish):
        """AI thực hiện nước đi."""
        if self.board.turn == chess.BLACK:  # Lượt của AI
            stockfish.set_fen_position(self.board.fen())
            best_move = stockfish.get_best_move()
            if best_move:
                self.board.push(chess.Move.from_uci(best_move))