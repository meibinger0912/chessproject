import chess
import chess.engine
import torch
import os
import random
import time
import numpy as np
from evaluator import EvaluatorNet
from ComputingAI import (
    evaluate_conventional, board_to_tensor, tensor_to_board, 
    king_moves, knight_moves, bishop_directions, rook_directions, queen_directions, pawn_captures,
    material_diff
)

# Biến toàn cục
turn = 1
winner = 0
position = torch.tensor([
    [-2, -3, -4, -5, -6, -4, -3, -2],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [2, 3, 4, 5, 6, 4, 3, 2]
])

class ChessDove:
    def __init__(self, depth=3, use_nn=True):
        """Khởi tạo AI cờ vua ChessDove sử dụng thư viện chess"""
        # Cấu hình tìm kiếm
        self.search_depth = depth
        self.use_neural_network = use_nn
        self.move_log = []
        self.position_history = {}
        self.move_count = 0  # Thêm biến đếm số nước đã đi
        
        # Tải mô hình đánh giá (nếu có)
        self.evaluator = None
        if self.use_neural_network:
            try:
                self.evaluator = EvaluatorNet()
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_eval.pt")
                if os.path.exists(model_path):
                    try:
                        self.evaluator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                        self.evaluator.eval()  # Đặt mô hình ở chế độ đánh giá
                        print("Loaded pre-trained evaluator model")
                        # Sử dụng độ sâu từ mô hình nếu có
                        if hasattr(self.evaluator, 'depth'):
                            self.search_depth = self.evaluator.depth
                    except Exception as e:
                        print(f"Error loading model: {e}")
                        print("Creating new model...")
                        create_initial_model()  # Tạo mô hình mới
                else:
                    print("No pre-trained evaluator model found. Creating new model...")
                    from evaluator import create_initial_model
                    create_initial_model()
            except Exception as e:
                print(f"Error initializing neural network: {str(e)}. Using default evaluation.")
                self.use_neural_network = False
        
        # Giá trị vật chất của các quân cờ
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Bảng vị trí cho các quân cờ
        self.piece_position_scores = self._init_piece_position_tables()
    
    def _init_piece_position_tables(self):
        """Khởi tạo bảng điểm vị trí cho các quân cờ"""
        from ComputingAI import init_piece_position_tables
        return init_piece_position_tables()
    
    def get_move(self, board, time_limit=10.0):
        """Tìm nước đi tốt nhất cho vị trí hiện tại"""
        # Ghi lại vị trí cho lịch sử
        self.move_log.append(board.fen())
        fen = board.fen().split()[0]  # Lấy phần FEN của vị trí
        self.position_history[fen] = self.position_history.get(fen, 0) + 1
        
        # Lấy danh sách nước đi hợp lệ
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Kiểm tra nếu có nước chiếu hết ngay lập tức
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()
        
        # Tính toán thời gian cho mỗi nước đi
        start_time = time.time()
        
        # Sử dụng tìm kiếm sâu dần (iterative deepening)
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        # Đảm bảo luôn có một nước đi (chọn ngẫu nhiên ban đầu)
        best_move = random.choice(legal_moves)
        
        try:
            for current_depth in range(1, self.search_depth + 1):
                # Kiểm tra thời gian sau mỗi độ sâu
                if time.time() - start_time > time_limit * 0.8:
                    break
                
                # Tìm kiếm ở độ sâu hiện tại
                if board.turn == chess.WHITE:
                    score, move = self.minimax(board, current_depth, float('-inf'), float('inf'), True, start_time, time_limit)
                    if score > best_score:
                        best_score = score
                        best_move = move
                else:
                    score, move = self.minimax(board, current_depth, float('-inf'), float('inf'), False, start_time, time_limit)
                    if score < best_score:
                        best_score = score
                        best_move = move
        
        except Exception as e:
            print(f"Error during search: {e}")
            # Nếu có lỗi, dùng nước đi ngẫu nhiên đã chọn ban đầu
        
        return best_move
    
    def minimax(self, board, depth, alpha, beta, maximizing, start_time, time_limit):
        """Tìm kiếm minimax với cắt tỉa alpha-beta và kiểm soát thời gian"""
        # Kiểm tra thời gian
        if time.time() - start_time > time_limit * 0.8:
            # Trả về đánh giá hiện tại và None (không có nước đi)
            return self.evaluate_position(board), None
        
        # Kiểm tra nút lá hoặc kết thúc game
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board), None
        
        # Sắp xếp nước đi để kiểm tra nước tấn công trước
        moves = list(board.legal_moves)
        
        # Ưu tiên các nước ăn quân
        capture_moves = []
        non_capture_moves = []
        
        for move in moves:
            if board.is_capture(move):
                capture_moves.append(move)
            else:
                non_capture_moves.append(move)
        
        # Xem xét nước tấn công trước
        sorted_moves = capture_moves + non_capture_moves
        
        best_move = None
        
        if maximizing:  # Lượt của MAX (trắng)
            best_score = float('-inf')
            for move in sorted_moves:
                board.push(move)
                score, _ = self.minimax(board, depth - 1, alpha, beta, False, start_time, time_limit)
                board.pop()
                
                # Thêm bonus nhỏ cho nước ăn quân
                if board.is_capture(move):
                    score += 0.001  # Bonus rất nhỏ để không ảnh hưởng đến minimax
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            
            return best_score, best_move
        
        else:  # Lượt của MIN (đen)
            best_score = float('inf')
            for move in sorted_moves:
                board.push(move)
                score, _ = self.minimax(board, depth - 1, alpha, beta, True, start_time, time_limit)
                board.pop()
                
                # Thêm bonus nhỏ cho nước ăn quân
                if board.is_capture(move):
                    score += 0.001  # Bonus rất nhỏ để không ảnh hưởng đến minimax
                
                if score < best_score:
                    best_score = score
                    best_move = move
                
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            
            return best_score, best_move
    
    def evaluate_position(self, board):
        """Đánh giá vị trí hiện tại"""
        # Kiểm tra kết thúc trận đấu
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            return 0
        
        # Kiểm tra lặp lại vị trí
        fen = board.fen().split()[0]  # Lấy phần FEN của vị trí
        if self.position_history.get(fen, 0) >= 3:
            return 0
        
        # Sử dụng mạng neural nếu có
        if self.use_neural_network and self.evaluator is not None:
            try:
                # Chuyển đổi bàn cờ thành tensor
                board_tensor = board_to_tensor(board)
                
                with torch.no_grad():
                    evaluation = self.evaluator(board_tensor.unsqueeze(0).unsqueeze(0).float())
                    eval_score = evaluation.item() * (1 if board.turn == chess.WHITE else -1)
            except Exception as e:
                print(f"Neural network evaluation error: {e}")
                # Fallback to conventional evaluation
                eval_score = evaluate_conventional(board, self.piece_values, self.piece_position_scores)
        else:
            # Conventional evaluation
            eval_score = evaluate_conventional(board, self.piece_values, self.piece_position_scores)
        
        # Phạt khi vua đi ra sớm trong giai đoạn đầu trận đấu
        if len(self.move_log) < 15:  # Trong 15 nước đầu
            # Kiểm tra vị trí vua
            white_king_square = board.king(chess.WHITE)
            black_king_square = board.king(chess.BLACK)
            
            if white_king_square is not None:
                white_king_rank = chess.square_rank(white_king_square)
                white_king_file = chess.square_file(white_king_square)
                
                # Khuyến khích nhập thành
                if white_king_square == chess.G1 or white_king_square == chess.C1:
                    eval_score += 50
                # Phạt khi vua rời vị trí ban đầu không phải để nhập thành
                elif white_king_square != chess.E1:
                    eval_score -= 100
            
            if black_king_square is not None:
                black_king_rank = chess.square_rank(black_king_square)
                black_king_file = chess.square_file(black_king_square)
                
                # Khuyến khích nhập thành
                if black_king_square == chess.G8 or black_king_square == chess.C8:
                    eval_score -= 50  # Lưu ý dấu trừ vì điểm âm có lợi cho đen
                # Phạt khi vua rời vị trí ban đầu không phải để nhập thành
                elif black_king_square != chess.E8:
                    eval_score += 100  # Lưu ý dấu cộng vì điểm dương bất lợi cho đen
        
        # Thêm phần thưởng cho việc tấn công
        if board.turn == chess.WHITE:
            # Thưởng cho các nước đi tấn công quân đối phương
            for move in board.legal_moves:
                if board.is_capture(move):
                    # Lấy giá trị quân bị ăn
                    to_square = move.to_square
                    captured_piece = board.piece_at(to_square)
                    if captured_piece:
                        # Thưởng bằng giá trị quân bị ăn
                        eval_score += self.piece_values[captured_piece.piece_type] * 0.1
        else:
            # Tương tự cho đen
            for move in board.legal_moves:
                if board.is_capture(move):
                    to_square = move.to_square
                    captured_piece = board.piece_at(to_square)
                    if captured_piece:
                        eval_score -= self.piece_values[captured_piece.piece_type] * 0.1
        
        # Thêm vào cuối hàm evaluate_position, ngay trước return:
        if random.random() < 0.1:  # 10% khả năng thêm nhiễu
            noise = random.uniform(-10, 10)
            eval_score += noise
        
        return eval_score

# Các hàm tương thích ngược cho codebase cũ
def get_legalmoves(position, turn, last_move=None, gamehis=None):
    """Trả về các nước đi hợp lệ cho tensor (tương thích ngược)"""
    # Khởi tạo gamehis nếu cần
    if gamehis is None:
        gamehis = []
    
    # Chuyển tensor thành bàn cờ
    board = tensor_to_board(position)
    
    # Đặt lượt đi
    board.turn = chess.WHITE if turn == 1 else chess.BLACK
    
    # Lấy các nước đi hợp lệ và chuyển sang định dạng ((x1,y1),(x2,y2))
    legal_moves = []
    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        
        from_rank = chess.square_rank(from_square)
        from_file = chess.square_file(from_square)
        
        to_rank = chess.square_rank(to_square)
        to_file = chess.square_file(to_square)
        
        legal_moves.append(((from_rank, from_file), (to_rank, to_file)))
    
    return legal_moves

def get_final_legal(position, turn, last_move=None, gamehis=None):
    return get_legalmoves(position, turn, last_move, gamehis)

def make_move(position, move, last_move=None):
   
    # Clone tensor để không thay đổi vị trí gốc
    new_position = position.clone()
    
    # Kiểm tra nước đi
    if not isinstance(move, tuple) or len(move) != 2:
        return position
    
    # Lấy tọa độ từ và đến
    from_pos, to_pos = move
    
    if not (isinstance(from_pos, tuple) and isinstance(to_pos, tuple)):
        return position
    
    from_rank, from_file = from_pos
    to_rank, to_file = to_pos
    
    # Kiểm tra biên bàn cờ
    if not (0 <= from_rank < 8 and 0 <= from_file < 8 and 0 <= to_rank < 8 and 0 <= to_file < 8):
        return position
    
    # Di chuyển quân cờ
    piece = position[from_rank, from_file].item()
    new_position[to_rank, to_file] = piece
    new_position[from_rank, from_file] = 0
    
    return new_position

def is_game_over(position, turn=None, last_move=None, gamehis=None):
    
    global winner
    
    # Xử lý tham số mặc định
    if turn is None:
        turn = 1
    if gamehis is None:
        gamehis = []
    
    # Chuyển tensor thành bàn cờ
    board = tensor_to_board(position)
    
    # Đặt lượt đi
    board.turn = chess.WHITE if turn == 1 else chess.BLACK
    
    # Kiểm tra kết thúc
    if board.is_game_over():
        if board.is_checkmate():
            winner = -turn  # Đối phương thắng
        else:
            winner = 0.5  # Hòa
        return True
    
    # Không kết thúc
    return False

def choose_best_move(position, depth=1, last_move=None, gamehis=None, evaluator=None):
    """Chọn nước đi tốt nhất cho vị trí hiện tại"""
    if gamehis is None:
        gamehis = []
    
    # Chuyển tensor thành bàn cờ python-chess
    board = tensor_to_board(position)
    board.turn = chess.WHITE  # Luôn là lượt của trắng trong ChessDove
    
    # Lấy danh sách nước đi hợp lệ từ python-chess
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return None
    
    # Tạo đối tượng ChessDove để đánh giá
    ai = ChessDove(depth=depth)
    
    # PHASE 1: CHECKMATE DETECTION 
    # Ưu tiên cao nhất cho nước chiếu hết ngay lập tức
    for move in legal_moves:
        board.push(move)
        if board.is_checkmate():
            from_square = move.from_square
            to_square = move.to_square
            from_rank = 7 - chess.square_rank(from_square)
            from_file = chess.square_file(from_square)
            to_rank = 7 - chess.square_rank(to_square)
            to_file = chess.square_file(to_square)
            print("Tìm thấy nước chiếu hết!")
            board.pop()
            return ((from_rank, from_file), (to_rank, to_file))
        board.pop()
    
    # PHASE 2: ORGANIZE MOVES BY CATEGORY
    # Phân loại nước đi theo mức độ ưu tiên
    check_capture_moves = []     # Nước vừa chiếu vừa ăn quân
    capture_king_threat_moves = [] # Nước ăn quân đang đe dọa vua
    capture_undefended_moves = [] # Nước ăn quân không được bảo vệ
    good_capture_moves = []      # Nước ăn quân có lợi (MVV-LVA)
    even_capture_moves = []      # Nước ăn quân ngang giá
    bad_capture_moves = []       # Nước ăn quân bất lợi nhưng có thể chấp nhận được
    check_moves = []             # Nước chiếu không ăn quân
    developing_moves = []        # Nước phát triển quân (di chuyển ra khỏi vị trí ban đầu)
    other_moves = []             # Các nước đi khác
    
    # Giá trị quân cờ
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320, 
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Đánh giá từng nước đi
    for move in legal_moves:
        # Kiểm tra xem nước đi có ăn quân không
        is_capture = board.is_capture(move)
        move_value = 0  # Giá trị cơ bản của nước đi
        
        # Lấy thông tin về quân cờ
        from_square = move.from_square
        to_square = move.to_square
        moving_piece = board.piece_at(from_square)
        
        if moving_piece is None:
            continue  # Skip invalid moves
            
        attacker_value = piece_values[moving_piece.piece_type]
        
        # Xác định giá trị của quân bị ăn (nếu có)
        victim_value = 0
        if is_capture:
            captured_piece = board.piece_at(to_square)
            if captured_piece:
                victim_value = piece_values[captured_piece.piece_type]
                move_value = victim_value - attacker_value/10  # MVV-LVA
        
        # Kiểm tra nước đi có tạo ra chiếu không
        board.push(move)
        gives_check = board.is_check()
        
        # Kiểm tra xem quân bị ăn có được bảo vệ không
        is_victim_defended = False
        victim_is_threat = False
        
        if is_capture:
            # Lưu lượt đi
            original_turn = board.turn
            
            # Thử đi ngược lại để xem quân vừa bị ăn có được bảo vệ không
            board.pop()  # Quay lại vị trí trước
            is_victim_defended = board.is_attacked_by(not moving_piece.color, to_square)
            
            # Kiểm tra xem quân bị ăn có đang đe dọa vua mình không
            victim_is_threat = False
            king_square = board.king(moving_piece.color)
            if king_square:
                attackers = board.attackers(not moving_piece.color, king_square)
                victim_is_threat = to_square in attackers
                
            # Đặt lại nước đi để tiếp tục phân loại
            board.push(move)
        
        # Di chuyển phát triển quân: quân rời vị trí ban đầu
        is_development = False
        if moving_piece.piece_type != chess.PAWN:
            start_rank = 0 if moving_piece.color == chess.BLACK else 7
            from_rank = chess.square_rank(from_square)
            if from_rank == start_rank and chess.square_rank(to_square) != start_rank:
                is_development = True
        
        # Phân loại nước đi dựa trên các tiêu chí
        if gives_check and is_capture:
            check_capture_moves.append((move, victim_value))
        elif is_capture:
            if victim_is_threat:
                capture_king_threat_moves.append((move, victim_value))
            elif not is_victim_defended:
                capture_undefended_moves.append((move, victim_value))  
            elif victim_value > attacker_value: 
                good_capture_moves.append((move, move_value))
            elif victim_value == attacker_value:
                even_capture_moves.append((move, move_value))
            else:
                # Chỉ xem xét ăn quân bất lợi nếu chênh lệch không quá lớn
                if victim_value >= attacker_value * 0.7:
                    bad_capture_moves.append((move, move_value))
        elif gives_check:
            check_moves.append((move, 50))  # Giá trị cơ bản cho nước chiếu
        elif is_development:
            developing_moves.append((move, 10))  # Giá trị cơ bản cho nước phát triển quân
        else:
            other_moves.append((move, 0))
        
        # Trả lại trạng thái bàn cờ
        board.pop()
    
    # PHASE 3: PRIORITIZE AND SORT MOVES
    
    # Sắp xếp các nhóm nước đi theo giá trị giảm dần
    check_capture_moves.sort(key=lambda x: x[1], reverse=True)
    capture_king_threat_moves.sort(key=lambda x: x[1], reverse=True)
    capture_undefended_moves.sort(key=lambda x: x[1], reverse=True)
    good_capture_moves.sort(key=lambda x: x[1], reverse=True)
    even_capture_moves.sort(key=lambda x: x[1], reverse=True)
    check_moves.sort(key=lambda x: x[1], reverse=True)
    developing_moves.sort(key=lambda x: x[1], reverse=True)
    
    # Ghép danh sách nước đi theo thứ tự ưu tiên
    prioritized_moves = []
    prioritized_moves.extend([m[0] for m in check_capture_moves])
    prioritized_moves.extend([m[0] for m in capture_king_threat_moves])
    prioritized_moves.extend([m[0] for m in capture_undefended_moves])
    prioritized_moves.extend([m[0] for m in good_capture_moves])
    prioritized_moves.extend([m[0] for m in check_moves])
    prioritized_moves.extend([m[0] for m in even_capture_moves])
    prioritized_moves.extend([m[0] for m in developing_moves])
    # Chỉ xem xét ăn quân bất lợi nếu không có lựa chọn tốt hơn
    if not prioritized_moves:
        prioritized_moves.extend([m[0] for m in bad_capture_moves])
    # Các nước đi khác được xem xét cuối cùng
    if not prioritized_moves:
        prioritized_moves.extend([m[0] for m in other_moves])
    
    # PHASE 4: DEPTH EVALUATION FOR BEST MOVE
    
    # Tìm nước đi tốt nhất dựa trên đánh giá sâu hơn
    best_move = None
    best_score = float('-inf')
    
    # Đảm bảo chọn nước đi mặc định nếu danh sách trống
    if prioritized_moves:
        best_move = prioritized_moves[0]
    else:
        # Nếu không có nước đi được phân loại, chọn ngẫu nhiên từ danh sách ban đầu
        best_move = random.choice(legal_moves) if legal_moves else None
    
    # Đánh giá các nước đi ưu tiên cao trước, rồi dần mở rộng nếu còn thời gian
    moves_to_evaluate = prioritized_moves[:min(10, len(prioritized_moves))]
    for move in moves_to_evaluate:
        board.push(move)
        # Kết hợp đánh giá từ neural network và đánh giá thông thường
        try:
            score = ai.evaluate_position(board)
            
            # Thêm bonus cho các nước đặc biệt để khuyến khích hành vi tốt
            if board.is_capture(move):
                # Thêm bonus lớn cho nước ăn quân
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    piece_type = captured_piece.piece_type
                    score += piece_values[piece_type] * 0.5
            
            if board.is_check():
                # Thưởng cho nước chiếu
                score += 100
                
            # Kiểm tra an toàn - không để quân có giá trị bị ăn miễn phí
            original_turn = board.turn
            board.turn = not board.turn  # Đổi lượt để kiểm tra đáp trả của đối phương
            opponent_captures = [m for m in board.legal_moves if board.is_capture(m)]
            max_potential_loss = 0
            for capture in opponent_captures:
                target = board.piece_at(capture.to_square)
                if target:
                    max_potential_loss = max(max_potential_loss, piece_values.get(target.piece_type, 0))
            
            # Nếu có khả năng mất quân có giá trị cao, trừ điểm đánh giá
            if max_potential_loss >= 300:  # Giá trị của mã/tượng trở lên
                score -= max_potential_loss * 0.8
            
            # Khôi phục lượt đi
            board.turn = original_turn
        except Exception as e:
            print(f"Lỗi khi đánh giá nước đi {move}: {e}")
            score = -100  # Điểm thấp cho nước đi gây lỗi
            
        board.pop()
        
        if score > best_score:
            best_score = score
            best_move = move
    
    # Nếu không tìm được nước đi tốt, chọn ngẫu nhiên
    if best_move is None and legal_moves:
        best_move = random.choice(legal_moves)
    
    # Chuyển đổi nước đi sang định dạng ChessDove
    if best_move:
        from_square = best_move.from_square
        to_square = best_move.to_square
        
        from_rank = 7 - chess.square_rank(from_square)
        from_file = chess.square_file(from_square)
        
        to_rank = 7 - chess.square_rank(to_square)
        to_file = chess.square_file(to_square)
        
        return ((from_rank, from_file), (to_rank, to_file))
    else:
        return None

def get_king_moves(position, turn, gamehis=None, last_move=None):
    """Trả về các nước đi hợp lệ cho quân vua (tương thích ngược)"""
    if gamehis is None:
        gamehis = []
    
    legal_moves = []
    kings = torch.nonzero((position * turn) == 6, as_tuple=False)
    
    if len(kings) == 0:
        print(f"[WARNING] No {('white' if turn == 1 else 'black')} king found on board!")
        return []
    
    king_pos = (kings[0][0].item(), kings[0][1].item())
    
    for dx, dy in king_moves:
        new_pos = (king_pos[0] + dx, king_pos[1] + dy)
        if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
            target = position[new_pos[0], new_pos[1]].item()
            if target * turn <= 0:  # Ô trống hoặc quân địch
                legal_moves.append((king_pos, new_pos))
    
    return legal_moves

def changeturn():
    """Đổi lượt (biến toàn cục)"""
    global turn
    turn *= -1