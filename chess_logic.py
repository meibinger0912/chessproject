import chess
from stockfish import Stockfish
from game_state import GameState
from chess_clock import ChessClock
import random

class ChessGame:
    def __init__(self, elo_rating, stockfish_path):
        self.board = chess.Board()
        self.stockfish = Stockfish(stockfish_path)
        self.stockfish.set_elo_rating(elo_rating)
        self.selected_square = None
        self.player_side = None
        self.attackable_squares = set()
        self.last_moved_piece = None
        self.game_state = GameState()
        self.game_result = None
        self.white_score = 0
        self.black_score = 0
        self.piece_values = {
            'P': 1,  # Pawn
            'N': 3,  # Knight
            'B': 3,  # Bishop
            'R': 5,  # Rook
            'Q': 9,  # Queen
            'K': 0   # King not counted in material score
        }
        self.clock = ChessClock()
        
        # Thêm hỗ trợ ChessDove
        self.chessdove_ai = None
        self.use_chessdove = False

    def set_player_side(self, player_side):
        """Set player side and start the clock"""
        self.player_side = player_side
        self.clock.start()
        # Khởi tạo AI đi trước nếu người chơi chọn đen
        if player_side == "black":
            print("AI (quân trắng) đi nước đầu tiên...")
            self.make_ai_move()

    def handle_click(self, row, col):
        """Handle player clicks and check game ending conditions."""
        # Check if game is over
        if self.board.is_game_over() or (hasattr(self, 'clock') and self.clock.game_over):
            return

        square = chess.square(col, row)
        player_color = chess.BLACK if self.player_side == "black" else chess.WHITE

        print(f"\nDebug click:")
        print(f"Tọa độ click: ({row}, {col})")
        print(f"Lượt hiện tại: {'Đen' if self.board.turn == chess.BLACK else 'Trắng'}")
        print(f"Bên người chơi: {self.player_side}")

        if self.board.turn == player_color:
            if self.selected_square is None:
                piece = self.board.piece_at(square)
                if piece and piece.color == player_color:
                    self.selected_square = square
                    print(f"Đã chọn quân: {piece.symbol()}")
            else:
                move = self.create_move(self.selected_square, square)
                if move and move in self.board.legal_moves:
                    moving_piece = self.board.piece_at(self.selected_square)
                    print(f"Di chuyển {moving_piece.symbol()} từ {chess.square_name(self.selected_square)} "
                          f"đến {chess.square_name(square)}")
                    
                    # In thông tin về nước đi đặc biệt
                    if self.is_castling_move(moving_piece, self.selected_square, square):
                        print("Nước nhập thành!")
                    elif self.is_promotion_move(moving_piece, square):
                        print("Nước phong cấp tốt thành hậu!")

                    self.board.push(move)
                    self.update_scores()
                    if hasattr(self, 'clock'):
                        self.clock.switch_player()
                    
                    # Kiểm tra game kết thúc sau mỗi nước đi
                    if not self.board.is_game_over():
                        self.make_ai_move()
                        self.update_scores()
                        if hasattr(self, 'clock'):
                            self.clock.switch_player()
                self.selected_square = None

    def create_move(self, from_square, to_square):
        """Tạo nước đi với xử lý nhập thành và phong cấp."""
        piece = self.board.piece_at(from_square)
        if not piece:
            return None

        # Xử lý phong cấp
        if piece.piece_type == chess.PAWN:
            if ((piece.color == chess.WHITE and chess.square_rank(to_square) == 7) or 
                (piece.color == chess.BLACK and chess.square_rank(to_square) == 0)):
                return chess.Move(from_square, to_square, promotion=chess.QUEEN)

        return chess.Move(from_square, to_square)

    def is_castling_move(self, piece, from_square, to_square):
        """Kiểm tra nước đi có phải là nhập thành không."""
        if not piece or piece.piece_type != chess.KING:
            return False
        return abs(chess.square_file(from_square) - chess.square_file(to_square)) > 1

    def is_promotion_move(self, piece, to_square):
        """Kiểm tra nước đi có phải là phong cấp không."""
        if not piece or piece.piece_type != chess.PAWN:
            return False
        return ((piece.color == chess.WHITE and chess.square_rank(to_square) == 7) or 
                (piece.color == chess.BLACK and chess.square_rank(to_square) == 0))

    def make_ai_move(self):
        """Make AI move and switch clock."""
        # Kiểm tra có phải lượt của AI không
        is_ai_turn = (self.player_side == "white" and self.board.turn == chess.BLACK) or \
                     (self.player_side == "black" and self.board.turn == chess.WHITE)

        if is_ai_turn:
            self.stockfish.set_fen_position(self.board.fen())
            best_move = self.stockfish.get_best_move()
            if best_move:
                self.board.push(chess.Move.from_uci(best_move))
                print(f"AI đã đi: {best_move}")
                # Cập nhật điểm và chuyển đồng hồ sau khi AI đi
                self.update_scores()
                if hasattr(self, 'clock'):
                    self.clock.last_move_by_ai = True  # Đánh dấu nước đi của AI
                    self.clock.switch_player()
                return True
        return False

    def get_legal_moves_for_selected_piece(self):
        """Lấy danh sách các nước đi hợp lệ cho quân cờ được chọn."""
        if self.selected_square is None:
            return []
        
        legal_moves = []
        for move in self.board.legal_moves:
            if move.from_square == self.selected_square:
                # Check if this move captures a piece
                is_capture = self.board.piece_at(move.to_square) is not None
                legal_moves.append((move, is_capture))
        return legal_moves

    def calculate_material_score(self, color):
        """Calculate material score for given color."""
        score = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            pieces = self.board.pieces(piece_type, color)
            score += len(pieces) * self.piece_values[chess.piece_symbol(piece_type).upper()]
        return score

    def update_scores(self):
        """Update scores after each move."""
        self.white_score = self.calculate_material_score(chess.WHITE)
        self.black_score = self.calculate_material_score(chess.BLACK)

    def check_game_state(self):
        """Kiểm tra trạng thái game."""
        if self.board.is_checkmate():
            winner = "Trắng" if self.board.turn == chess.BLACK else "Đen"
            self.game_result = f"Chiếu hết! {winner} thắng!"
            return True
        elif self.board.is_stalemate():
            self.game_result = "Hòa cờ do bế tắc!"
            return True
        elif self.board.is_insufficient_material():
            self.game_result = "Hòa cờ do thiếu quân!"
            return True
        elif self.board.is_fifty_moves():
            self.game_result = "Hòa cờ do 50 nước không ăn quân!"
            return True
        elif self.board.is_repetition():
            self.game_result = "Hòa cờ do lặp lại nước đi!"
            return True
        return False

    def is_game_over(self):
        """Check if game is over including time limit"""
        return (self.board.is_game_over() or 
                (hasattr(self, 'clock') and self.clock.game_over))

    def get_game_result(self):
        """Get the game result message"""
        # First check time-based ending
        if hasattr(self, 'clock') and self.clock.game_over:
            return self.clock.result_message

        # Then check regular chess endings
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            return f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            return "Draw by stalemate!"
        elif self.board.is_insufficient_material():
            return "Draw by insufficient material!"
        elif self.board.is_repetition():
            return "Draw by repetition!"
        
        return None

    def get_ai_move(self):
        """Lấy nước đi từ AI."""
        if self.use_chessdove and self.chessdove_ai:
            # Sử dụng ChessDove AI
            try:
                move = self.chessdove_ai.get_move(self.board)
                return move
            except Exception as e:
                print(f"Lỗi ChessDove AI: {e}")
                # Trả về nước đi ngẫu nhiên nếu có lỗi
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    return random.choice(legal_moves)
                return None
        else:
            # Sử dụng Stockfish
            best_move = self.stockfish.get_best_move()
            if best_move:
                return chess.Move.from_uci(best_move)
            return None