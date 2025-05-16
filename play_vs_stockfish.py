import chess
import chess.engine
import torch
import os
import time
import sys
import random
import numpy as np
from datetime import datetime
from ChessDove import position as initial_position, make_move, choose_best_move, get_legalmoves
from evaluator import EvaluatorNet

# Biến chế độ debug
DEBUG_MODE = True  # Đặt thành False trong môi trường sản xuất

# Tìm kiếm model_weights.pt ở nhiều vị trí
possible_model_paths = [
    "model_weights.pt",
    "best_eval.pt",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.pt"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_eval.pt"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_weights.pt")
]

evaluator = None
for model_path in possible_model_paths:
    if os.path.exists(model_path):
        try:
            evaluator = EvaluatorNet()
            evaluator.load_state_dict(torch.load(model_path))
            evaluator.eval()
            print(f"Loaded evaluation model from {model_path}")
            break
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")

if evaluator is None:
    print("Warning: No evaluation model found. Using default evaluation.")

def debug_log(message):
    """In thông báo debug nếu ở chế độ debug"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

# Thiết lập logging
LOG_FILE = "training_log.txt"
DATA_DIR = "training_data"
os.makedirs(DATA_DIR, exist_ok=True)

def log_message(message):
    """Ghi log với timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

# Hàm chuyển đổi giữa biểu diễn bàn cờ
def tensor_to_fen(position):
    """Chuyển tensor 8x8 sang chuỗi FEN"""
    piece_map = {1:'P', 2:'R', 3:'N', 4:'B', 5:'Q', 6:'K',
                -1:'p', -2:'r', -3:'n', -4:'b', -5:'q', -6:'k', 0:' '}
    fen = ''
    for i in range(8):
        empty = 0
        for j in range(8):
            val = position[i][j].item()
            if val == 0:
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += piece_map[val]
        if empty > 0:
            fen += str(empty)
        if i < 7:
            fen += '/'
    
    # Thêm thông tin thêm: turn, castling rights, en passant, halfmove, fullmove
    # Mặc định cho đơn giản
    fen += ' w - - 0 1'
    return fen

def fen_to_tensor(fen):
    """Chuyển chuỗi FEN sang tensor 8x8"""
    piece_map = {'P':1, 'R':2, 'N':3, 'B':4, 'Q':5, 'K':6,
                'p':-1, 'r':-2, 'n':-3, 'b':-4, 'q':-5, 'k':-6}
    
    board_part = fen.split()[0]
    rows = board_part.split('/')
    
    tensor = torch.zeros((8, 8), dtype=torch.float32)
    
    for i, row in enumerate(rows):
        j = 0
        for char in row:
            if char.isdigit():
                j += int(char)
            else:
                tensor[i][j] = piece_map[char]
                j += 1
                
    return tensor

def uci_to_move(uci_move, board_tensor=None):
    """Chuyển nước đi UCI sang định dạng ChessDove"""
    file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    
    from_file = file_map[uci_move[0]]
    from_rank = 7 - (int(uci_move[1]) - 1)  # Sửa công thức chuyển đổi
    to_file = file_map[uci_move[2]]
    to_rank = 7 - (int(uci_move[3]) - 1)
    
    return ((from_rank, from_file), (to_rank, to_file))

def move_to_uci(move):
    """Chuyển nước đi ChessDove sang UCI"""
    file_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    
    from_rank, from_file = move[0]
    to_rank, to_file = move[1]
    
    # Sửa công thức chuyển đổi
    uci = file_map[from_file] + str(8 - from_rank) + file_map[to_file] + str(8 - to_rank)
    return uci

# Sửa lại hàm play_games trong play_vs_stockfish.py để xử lý last_move đúng
def play_games(num_games=100, stockfish_depth=5, stockfish_path="D:\\Trí tuệ nhân tạo\\chess project\\stockfish\\stockfish.exe"):
    """Chơi num_games trận với Stockfish và thu thập dữ liệu huấn luyện"""
    
    # Kiểm tra stockfish có tồn tại
    if not os.path.exists(stockfish_path):
        log_message(f"Error: Stockfish not found at {stockfish_path}")
        log_message("Please download stockfish and update the path")
        return
    
    # Khởi tạo Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Skill Level": stockfish_depth})
        log_message(f"Stockfish initialized at depth/skill level {stockfish_depth}")
    except Exception as e:
        log_message(f"Error initializing Stockfish: {e}")
        return
    
    # Thống kê
    stats = {"wins": 0, "draws": 0, "losses": 0}
    positions_data = []  # Danh sách (position, value) để huấn luyện
    
    log_message(f"Starting {num_games} games against Stockfish")
    
    for game_idx in range(num_games):
        log_message(f"Game {game_idx + 1}/{num_games}")
        
        # Khởi tạo bàn cờ mới
        pos = initial_position.clone()
        board = chess.Board()
        
        # Lưu trữ các vị trí của game hiện tại
        game_positions = []
        last_move = None  # Thêm biến theo dõi nước đi cuối cùng
        game_history = []  # Thêm biến theo dõi lịch sử game
        
        # Đấu trận
        move_count = 0
        while not board.is_game_over():
            move_count += 1
            log_message(f"Game {game_idx + 1}, Move {move_count}")
            
            # Đặt giới hạn số nước đi để tránh game quá dài
            if move_count > 200:
                log_message("Game too long, forcing a draw")
                stats["draws"] += 1
                break
            
            # Lượt của ChessDove (luôn đi trắng để đơn giản)
            current_fen = board.fen()
            
            # Tính nước đi tốt nhất bằng minimax + neural network
            try:
                debug_log(f"Calling choose_best_move with depth=1") 
                debug_log(f"Position shape: {pos.shape}, Last move: {last_move}")
                
                # Truyền evaluator nếu có
                if evaluator:
                    move = choose_best_move(pos, depth=1, last_move=last_move, gamehis=game_history, evaluator=evaluator)
                else:
                    move = choose_best_move(pos, depth=1, last_move=last_move, gamehis=game_history)
                
                # Thêm đoạn code này để làm đa dạng nước đi trong quá trình huấn luyện
                if random.random() < 0.2:  # 20% cơ hội chọn ngẫu nhiên
                    from ChessDove import get_legalmoves
                    legal_moves = get_legalmoves(pos, True, last_move)
                    if legal_moves:
                        move = random.choice(legal_moves)
                        debug_log(f"Random move selected: {move}")
                        
                if move is None:
                    log_message("No legal moves found. Ending game.")
                    break
                debug_log(f"ChessDove chose move: {move}")
            except Exception as e:
                log_message(f"Error in choose_best_move: {str(e)}")
                debug_log(f"Exception details: {type(e).__name__}: {str(e)}")
                
                # Thử với độ sâu thấp hơn
                try:
                    log_message("Trying with lower depth...")
                    move = choose_best_move(pos, depth=1)
                    if move is None:
                        log_message("Still no legal moves. Ending game.")
                        break
                except Exception as e2:
                    log_message(f"Second error: {str(e2)}")
                    log_message("Could not recover. Ending game.")
                    break
            
            # Thêm vào trước khi thực hiện nước đi của AI
            legal_moves_uci = [move.uci() for move in board.legal_moves]
            debug_log(f"Legal moves according to python-chess: {legal_moves_uci}")

            # Convert move sang UCI và áp dụng
            uci_move = move_to_uci(move)
            chess_move = chess.Move.from_uci(uci_move)

            # Kiểm tra nước đi hợp lệ một cách nghiêm ngặt
            if chess_move not in board.legal_moves:
                debug_log(f"AI attempted illegal move: {uci_move}")
                # Sửa: Sử dụng các nước đi hợp lệ từ python-chess thay vì ChessDove
                if board.legal_moves:
                    chess_move = random.choice(list(board.legal_moves))
                    uci_move = chess_move.uci()
                    move = uci_to_move(uci_move)
                    debug_log(f"Replaced with random legal move: {uci_move}")
                else:
                    log_message("No legal moves available. Ending game.")
                    break
            
            # Lưu vị trí hiện tại và nước đi
            game_positions.append((pos.clone(), None))  # Value sẽ được cập nhật sau
            game_history.append(pos.clone())  # Cập nhật lịch sử game
            
            # Thực hiện nước đi
            board.push(chess_move)
            pos = make_move(pos, move)
            last_move = move  # Cập nhật nước đi cuối cùng
            
            # Kiểm tra kết thúc
            if board.is_game_over():
                break
                
            # Lượt của Stockfish
            try:
                # Xóa tham số timeout vì SimpleEngine.play() không chấp nhận nó
                result = engine.play(board, chess.engine.Limit(time=0.1))
                # Sau khi Stockfish thực hiện nước đi
                board.push(result.move)
                log_message(f"Stockfish played: {result.move.uci()}")

                # *** CẬP NHẬT QUAN TRỌNG: Tạo lại pos từ trạng thái bàn cờ hiện tại
                pos = fen_to_tensor(board.fen())
                last_move = uci_to_move(result.move.uci(), pos)
                game_history.append(pos.clone())
            except Exception as e:
                log_message(f"Error during Stockfish's turn: {str(e)}")
                log_message("Ending game due to Stockfish error")
                break
            
            # Cập nhật tensor pos và thông tin nước đi cuối cùng từ FEN mới
            pos = fen_to_tensor(board.fen())
            last_move = uci_to_move(result.move.uci(), pos)  # Cập nhật nước đi cuối cùng
            
        # Game kết thúc, xử lý kết quả
        result_value = 0
        if board.is_checkmate():
            if board.turn:  # Đen thắng (ChessDove thua)
                stats["losses"] += 1
                result_value = -1
            else:  # Trắng thắng (ChessDove thắng)
                stats["wins"] += 1
                result_value = 1
        else:
            stats["draws"] += 1
            result_value = 0
            
        log_message(f"Game {game_idx + 1} result: {result_value}")
        
        # Cập nhật giá trị cho tất cả vị trí trong game
        for i, (position, _) in enumerate(game_positions):
            # Discount factor để giá trị game giảm dần về quá khứ
            # Càng gần kết quả cuối cùng, giá trị càng cao
            discount = 0.95 ** (len(game_positions) - i - 1)
            positions_data.append((position, result_value * discount))
            
        # In thống kê
        log_message(f"Current stats - Wins: {stats['wins']}, Draws: {stats['draws']}, Losses: {stats['losses']}")
        
        # Lưu dữ liệu định kỳ
        if (game_idx + 1) % 10 == 0:
            save_training_data(positions_data, game_idx + 1)
            
    engine.quit()
    return positions_data, stats

# Sửa hàm save_training_data()

def save_training_data(positions_data, game_idx):
    """Lưu dữ liệu huấn luyện thu thập được"""
    if not positions_data:
        log_message("No positions data to save")
        return
        
    try:
        states = torch.stack([pos for pos, _ in positions_data])
        values = torch.tensor([val for _, val in positions_data])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Lưu dữ liệu
        save_path_states = os.path.join(DATA_DIR, f"states_{timestamp}_game{game_idx}.pt")
        save_path_values = os.path.join(DATA_DIR, f"values_{timestamp}_game{game_idx}.pt")
        
        torch.save(states, save_path_states)
        torch.save(values, save_path_values)
        
        log_message(f"Saved {len(positions_data)} training examples to {DATA_DIR}")
        log_message(f"States saved to: {save_path_states}")
        log_message(f"Values saved to: {save_path_values}")
    except Exception as e:
        log_message(f"Error saving training data: {str(e)}")

# Sửa phần cuối file để kiểm tra đường dẫn Stockfish

if __name__ == "__main__":
    # Cài đặt cho đấu với Stockfish
    num_games = 1000  
    stockfish_depth = 5  
    log_message(f"Starting training with {num_games} games at depth {stockfish_depth}")
    
    # Thử nhiều đường dẫn có thể
    possible_paths = [
        r"D:\Trí tuệ nhân tạo\chess project\stockfish\stockfish.exe",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "stockfish", "stockfish.exe"),
        "stockfish.exe"
    ]
    
    stockfish_path = None
    for path in possible_paths:
        if os.path.exists(path):
            stockfish_path = path
            log_message(f"Found Stockfish at: {stockfish_path}")
            break
    
    if stockfish_path is None:
        log_message("Error: Could not find Stockfish executable")
        log_message("Please download Stockfish and place it in the stockfish directory")
        sys.exit(1)
    
    log_message("Starting training data collection")
    positions_data, stats = play_games(num_games, stockfish_depth, stockfish_path)
    log_message(f"Data collection complete. Final stats: {stats}")
    
    # Lưu toàn bộ dữ liệu huấn luyện
    save_training_data(positions_data, num_games)
    log_message(f"All training data saved to {DATA_DIR}")

def get_final_legal(board_tensor, turn, last_move=None, game_history=None):
    """Tạo danh sách nước đi hợp lệ cuối cùng sau khi lọc"""
    # Logic cũ (nếu có)
    # Nếu không có logic cũ, hãy thêm mã giả để tránh lỗi:
    moves = []  # Khởi tạo một danh sách rỗng
    
    # Thêm code để sinh nước đi hợp lệ ở đây
    # ...
    
    print(f"[DEBUG] [get_final_legal] Số nước đi sinh ra: {len(moves)}")
    return moves

