import chess
import torch
import math
import numpy as np
import random

# Định nghĩa hướng di chuyển cho các quân cờ
king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
bishop_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
rook_directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
queen_directions = bishop_directions + rook_directions
pawn_captures = {1: [(-1, -1), (-1, 1)], -1: [(1, -1), (1, 1)]}

def init_piece_position_tables():
    """Khởi tạo bảng điểm vị trí cho các quân cờ"""
    return {
        chess.PAWN: [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ],
        chess.KNIGHT: [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ],
        chess.BISHOP: [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ],
        chess.ROOK: [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ],
        chess.QUEEN: [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ],
        chess.KING: [
            -50,-40,-30,-20,-20,-30,-40,-50,
            -30,-20,-10,-10,-10,-10,-20,-30,
            -30,-10,-20,-20,-20,-20,-10,-30,
            -30,-10,-20,-20,-20,-20,-10,-30,
            -30,-10,-20,-20,-20,-20,-10,-30,
            -30,-20,-20,-20,-20,-20,-20,-30,
            -30,-40,-40,-40,-40,-40,-40,-30,  
            -20,-10,-10,-10,-10,-10,-10,-20   
        ]
    }

def evaluate_conventional(board, piece_values, piece_position_scores):
    """Conventional chess position evaluation"""
    if board.is_game_over():
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        return 0  # Draw
    
    score = 0
    
    # Count material with piece-square tables
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        value = piece_values[piece.piece_type]
        
        # Get positional value
        if piece.piece_type in piece_position_scores:
            square_idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
            position_value = piece_position_scores[piece.piece_type][square_idx]
            value += position_value
        
        # Add to score (positive for white, negative for black)
        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value
    
    # Mobility evaluation
    if board.turn == chess.WHITE:
        # Store current turn
        turn = board.turn
        
        # Count white mobility
        white_mobility = len(list(board.legal_moves))
        
        # Switch turn to black and count mobility
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        
        # Restore turn
        board.turn = turn
        
        # Add mobility differential to score
        score += (white_mobility - black_mobility) * 5
    else:
        # Store current turn
        turn = board.turn
        
        # Count black mobility
        black_mobility = len(list(board.legal_moves))
        
        # Switch turn to white and count mobility
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        
        # Restore turn
        board.turn = turn
        
        # Add mobility differential to score
        score -= (white_mobility - black_mobility) * 5
    
    # Đếm số nước đi đã thực hiện
    move_count = len(list(board.move_stack))
    
    # Kiểm tra vị trí vua trong giai đoạn đầu
    if move_count < 15:  # 15 nước đầu tiên
        # Vị trí ban đầu của vua trắng và đen
        white_king_start = chess.E1
        black_king_start = chess.E8
        
        # Vị trí nhập thành của vua
        white_king_castled = [chess.G1, chess.C1]  # Nhập thành ngắn/dài
        black_king_castled = [chess.G8, chess.C8]  # Nhập thành ngắn/dài
        
        # Tìm vị trí hiện tại của vua
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        # Phạt nặng nếu vua di chuyển mà không phải để nhập thành
        if white_king_square != white_king_start and white_king_square not in white_king_castled:
            score -= 200  # Giảm điểm cho bên trắng
        
        if black_king_square != black_king_start and black_king_square not in black_king_castled:
            score += 200  # Tăng điểm cho bên trắng (giảm cho bên đen)
        
        # Thưởng cho việc nhập thành
        if white_king_square in white_king_castled:
            score += 50
        
        if black_king_square in black_king_castled:
            score -= 50
    
    # Thêm đánh giá cho việc tấn công
    # Đánh giá các quân đang bị tấn công
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        # Kiểm tra xem quân này có đang bị tấn công không
        is_attacked_by_opponent = board.is_attacked_by(not piece.color, square)
        if is_attacked_by_opponent:
            # Thưởng/phạt cho việc tấn công quân đối phương
            attack_value = piece_values[piece.piece_type] * 0.2
            if piece.color == chess.WHITE:
                score -= attack_value  # Quân trắng đang bị tấn công
            else:
                score += attack_value  # Quân đen đang bị tấn công
    
    return score

def board_to_tensor(board):
    """Chuyển đổi bàn cờ từ thư viện chess sang tensor"""
    tensor = torch.zeros((8, 8), dtype=torch.float32)
    
    # Map các loại quân cờ sang giá trị tensor
    piece_values = {
        'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6,
        'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[rank][file] = piece_values[piece.symbol()]
    
    return tensor

def tensor_to_board(tensor):
    """Chuyển đổi tensor sang bàn cờ thư viện chess"""
    board = chess.Board(fen=None)  # Bàn trống
    board.clear_board()
    
    # Map giá trị tensor sang các loại quân cờ
    piece_map = {
        1: chess.Piece(chess.PAWN, chess.WHITE),
        2: chess.Piece(chess.ROOK, chess.WHITE),
        3: chess.Piece(chess.KNIGHT, chess.WHITE),
        4: chess.Piece(chess.BISHOP, chess.WHITE),
        5: chess.Piece(chess.QUEEN, chess.WHITE),
        6: chess.Piece(chess.KING, chess.WHITE),
        -1: chess.Piece(chess.PAWN, chess.BLACK),
        -2: chess.Piece(chess.ROOK, chess.BLACK),
        -3: chess.Piece(chess.KNIGHT, chess.BLACK),
        -4: chess.Piece(chess.BISHOP, chess.BLACK),
        -5: chess.Piece(chess.QUEEN, chess.BLACK),
        -6: chess.Piece(chess.KING, chess.BLACK)
    }
    
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            piece_value = int(tensor[rank][file].item())
            if piece_value != 0:
                board.set_piece_at(square, piece_map[piece_value])
    
    return board

def material_equ(position):
    """Tính điểm vật chất từ tensor"""
    score = 0
    
    value_map = {1: 100, 2: 500, 3: 320, 4: 330, 5: 900, 6: 20000}
    
    for i in range(8):
        for j in range(8):
            piece = position[i][j].item()
            if piece != 0:
                value = value_map[abs(int(piece))]
                score += value if piece > 0 else -value
    
    return score

def material_diff(position_history):
    """Tính hiệu số vật chất giữa hai bên"""
    if not position_history or len(position_history) == 0:
        return 0  # Trả về 0 thay vì None nếu không có lịch sử
    
    pos = position_history[-1]
    material_values = {
        1: 1,    # Tốt
        2: 5,    # Xe
        3: 3,    # Mã
        4: 3,    # Tượng
        5: 9,    # Hậu
        6: 100   # Vua
    }
    
    white_material = 0
    black_material = 0
    
    for i in range(8):
        for j in range(8):
            piece_value = int(pos[i, j].item())
            if piece_value > 0:
                white_material += material_values[piece_value]
            elif piece_value < 0:
                black_material += material_values[-piece_value]
    
    return white_material - black_material

def evaluate_against_stockfish(num_games, stockfish_depth, stockfish_path, time_limit=5.0):
    """Đấu với Stockfish để đánh giá sức mạnh"""
    import chess.engine
    from ChessDove import ChessDove
    
    wins, losses, draws = 0, 0, 0
    
    # Khởi tạo Stockfish engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Skill Level": stockfish_depth})
        print(f"Khởi tạo Stockfish với độ sâu {stockfish_depth}")
    except Exception as e:
        print(f"Lỗi khởi tạo Stockfish: {e}")
        return 0, 0, 0
    
    try:
        # Đấu các trận đánh giá
        for i in range(num_games):
            print(f"Trận {i+1}/{num_games}")
            
            # Khởi tạo bàn cờ mới
            board = chess.Board()
            
            # Đấu trận
            move_count = 0
            
            # ChessDove đi quân trắng
            ai = ChessDove(depth=3)
            
            while not board.is_game_over():
                move_count += 1
                
                # Giới hạn số nước đi
                if move_count > 100:
                    print("Trận đấu quá dài, coi như hòa")
                    draws += 1
                    break
                
                if board.turn == chess.WHITE:  # Lượt của ChessDove
                    try:
                        # Tìm nước đi tốt nhất
                        chess_move = ai.get_move(board, time_limit)
                        if chess_move is None:
                            print("ChessDove không tìm được nước đi hợp lệ")
                            losses += 1
                            break
                        
                        # Thực hiện nước đi
                        board.push(chess_move)
                        
                    except Exception as e:
                        print(f"Lỗi trong lượt của ChessDove: {e}")
                        losses += 1
                        break
                
                else:  # Lượt của Stockfish
                    try:
                        # Giới hạn thời gian của Stockfish với độ sâu tương ứng
                        result = engine.play(board, chess.engine.Limit(time=time_limit))
                        board.push(result.move)
                        
                    except Exception as e:
                        print(f"Lỗi trong lượt của Stockfish: {e}")
                        wins += 1
                        break
            
            # Kiểm tra kết quả
            if board.is_game_over():
                result = board.result()
                if result == "1-0":  # ChessDove thắng
                    wins += 1
                    print("ChessDove thắng!")
                elif result == "0-1":  # Stockfish thắng
                    losses += 1
                    print("Stockfish thắng!")
                else:  # Hòa
                    draws += 1
                    print("Hòa!")
            
            print(f"Kết thúc trận {i+1}, tỉ số: Thắng {wins}, Thua {losses}, Hòa {draws}")
    
    finally:
        # Đảm bảo đóng engine khi hoàn thành
        try:
            engine.quit()
        except:
            pass
    
    return wins, losses, draws

def calculate_elo(wins, losses, draws, opponent_elo=2200):
    """Tính điểm Elo dựa trên kết quả đấu"""
    total_games = wins + losses + draws
    if total_games == 0:
        return 1500  # Điểm Elo mặc định
    
    score = (wins + 0.5 * draws) / total_games
    
    # Công thức ước tính Elo
    # Rd = -400 * log10(1/S - 1)
    # Elo = opponent_elo + Rd
    
    if score == 0:
        score = 0.01  # Tránh chia cho 0
    elif score == 1:
        score = 0.99  # Tránh logarithm của 0
    
    rating_diff = -400 * math.log10(1/score - 1)
    elo = opponent_elo + rating_diff
    
    return elo

def save_elo_history(elo, wins, losses, draws, stockfish_depth):
    """Lưu lịch sử đánh giá Elo"""
    import json
    import os
    from datetime import datetime
    
    # Tạo dictionary chứa thông tin
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "date": now,
        "elo": float(elo),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "stockfish_depth": stockfish_depth,
        "total_games": wins + losses + draws
    }
    
    # Đọc lịch sử hiện có hoặc tạo mới
    history_file = "elo_history.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except:
            history = []
    else:
        history = []
    
    # Thêm kết quả mới vào lịch sử
    history.append(data)
    
    # Lưu lại
    try:
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        print("Đã lưu lịch sử Elo")
    except Exception as e:
        print(f"Không thể lưu lịch sử Elo: {e}")

def is_king_safe(position, turn):
    """Kiểm tra xem vua có an toàn không (không bị chiếu)"""
    # Đơn giản hóa: giả sử vua luôn an toàn trong phiên bản này
    return True