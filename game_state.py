import chess

class GameState:
    def __init__(self):
        self.current_turn = chess.WHITE
        self.moves_history = []
        self.last_move = None
        self.player_side = None

    def switch_turn(self):
        """Chuyển lượt chơi."""
        self.current_turn = not self.current_turn

    def is_white_turn(self):
        """Kiểm tra có phải lượt của quân trắng không."""
        return self.current_turn == chess.WHITE

    def set_player_side(self, side):
        """Thiết lập bên cho người chơi"""
        self.player_side = side

    def get_current_color(self):
        """Lấy màu quân đang đến lượt"""
        return "Trắng" if self.current_turn == chess.WHITE else "Đen"

    def add_move(self, move, piece):
        """Thêm nước đi vào lịch sử"""
        move_info = {
            "move": move,
            "piece": piece,
            "color": "Trắng" if self.current_turn == chess.WHITE else "Đen",
            "number": len(self.moves_history) + 1
        }
        self.moves_history.append(move_info)
        self.last_move = move_info
        print(f"Nước {move_info['number']}: {move_info['color']} di chuyển {piece} từ "
              f"{chess.square_name(move.from_square)} đến {chess.square_name(move.to_square)}")
        self.switch_turn()