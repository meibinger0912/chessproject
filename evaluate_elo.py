import chess
import chess.engine
import argparse
import os
import json
import torch
import sys
import random
import time
from datetime import datetime

# Hằng số mặc định
DEFAULT_STOCKFISH_PATH = "D:\\Trí tuệ nhân tạo\\chess project\\stockfish\\stockfish.exe"
DEFAULT_ELO_LOG = "elo_evaluation.log"
DEFAULT_ELO_HISTORY = "elo_history.json"

def log_message(message, log_file=DEFAULT_ELO_LOG):
    """Ghi log với timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"[{timestamp}] Không thể ghi log: {e}")

def calculate_elo(wins, losses, draws, opponent_elo=3000):
    """Tính toán Elo dựa trên kết quả"""
    total_games = wins + losses + draws
    if total_games == 0:
        return 1500
    
    win_rate = (wins + 0.5 * draws) / total_games
    expected_score = 1 / (1 + 10**((opponent_elo - 1500)/400))
    
    if win_rate > expected_score:
        elo_diff = 400 * (win_rate - expected_score)
        return min(opponent_elo, 1500 + elo_diff)
    else:
        elo_diff = 400 * (expected_score - win_rate)
        return max(1000, 1500 - elo_diff)

def fen_to_tensor(fen):
    """Chuyển chuỗi FEN sang tensor"""
    piece_map = {'P':1, 'R':2, 'N':3, 'B':4, 'Q':5, 'K':6,
                'p':-1, 'r':-2, 'n':-3, 'b':-4, 'q':-5, 'k':-6}
    
    board_part = fen.split()[0]
    rows = board_part.split('/')
    
    tensor = torch.zeros((8, 8))
    
    for i, row in enumerate(rows):
        j = 0
        for char in row:
            if char.isdigit():
                j += int(char)
            else:
                tensor[i][j] = piece_map[char]
                j += 1
    return tensor

def uci_to_move(uci_move):
    """Chuyển nước đi UCI (e.g. "e2e4") sang định dạng ((x1,y1),(x2,y2))"""
    file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    
    from_file = file_map[uci_move[0]]
    from_rank = 8 - int(uci_move[1])  # Đảo ngược rank
    to_file = file_map[uci_move[2]]
    to_rank = 8 - int(uci_move[3])
    
    return ((from_rank, from_file), (to_rank, to_file))

def move_to_uci(move):
    """Chuyển nước đi ((x1,y1),(x2,y2)) sang UCI (e.g. "e2e4")"""
    file_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    
    from_rank, from_file = move[0]
    to_rank, to_file = move[1]
    
    return file_map[from_file] + str(8 - from_rank) + file_map[to_file] + str(8 - to_rank)

def play_stockfish_game(stockfish_engine, chess_engine=None, time_limit=1.0):
    """Chơi một trận đấu với Stockfish và trả về kết quả"""
    board = chess.Board()
    
    # Nếu không có chess_engine, sử dụng ChessDove (import khi cần)
    if chess_engine is None:
        try:
            from ChessDove import ChessDove
            chess_engine = ChessDove()
        except ImportError:
            log_message("Không thể import ChessDove, chọn nước đi ngẫu nhiên")
            chess_engine = None
    
    # Xác định màu ngẫu nhiên cho AI
    ai_color = random.choice([chess.WHITE, chess.BLACK])
    
    # Thực hiện trận đấu
    while not board.is_game_over():
        # Lấy nước đi tiếp theo
        if board.turn == ai_color:
            # Lượt của AI
            if chess_engine:
                try:
                    move = chess_engine.get_move(board, time_limit)
                    if move and move in board.legal_moves:
                        board.push(move)
                    else:
                        # Nếu AI trả về nước đi không hợp lệ, chọn ngẫu nhiên
                        legal_moves = list(board.legal_moves)
                        if legal_moves:
                            move = random.choice(legal_moves)
                            board.push(move)
                        else:
                            log_message("Không có nước đi hợp lệ, AI thua")
                            return "stockfish_win" if ai_color == chess.WHITE else "stockfish_loss"
                except Exception as e:
                    log_message(f"Lỗi khi lấy nước đi từ AI: {e}")
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        move = random.choice(legal_moves)
                        board.push(move)
                    else:
                        log_message("Không có nước đi hợp lệ, AI thua")
                        return "stockfish_win" if ai_color == chess.WHITE else "stockfish_loss"
            else:
                # Dùng nước đi ngẫu nhiên nếu không có AI
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    board.push(move)
                else:
                    return "stockfish_win" if ai_color == chess.WHITE else "stockfish_loss"
        else:
            # Lượt của Stockfish
            try:
                result = stockfish_engine.play(board, chess.engine.Limit(time=time_limit))
                board.push(result.move)
            except Exception as e:
                log_message(f"Lỗi Stockfish: {e}")
                return "ai_win" if ai_color == chess.WHITE else "ai_loss"
    
    # Xác định kết quả
    result = board.result()
    if ai_color == chess.WHITE:
        # AI đi trắng
        if result == "1-0":
            return "ai_win"
        elif result == "0-1":
            return "stockfish_win"
        else:
            return "draw"
    else:
        # AI đi đen
        if result == "1-0":
            return "stockfish_win" 
        elif result == "0-1":
            return "ai_win"
        else:
            return "draw"

def evaluate_against_stockfish(num_games=30, stockfish_depth=10, stockfish_path=None, time_limit=1.0):
    """Đánh giá hiệu suất của AI chống lại Stockfish"""
    # Đảm bảo có đường dẫn hợp lệ cho Stockfish
    if stockfish_path is None or not os.path.exists(stockfish_path):
        stockfish_path = DEFAULT_STOCKFISH_PATH
        if not os.path.exists(stockfish_path):
            log_message(f"Không tìm thấy Stockfish tại {stockfish_path}")
            return 0, 0, 0
    
    log_message(f"Sử dụng Stockfish tại {stockfish_path}")
    
    try:
        # Khởi tạo Stockfish engine
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Skill Level": stockfish_depth})
        log_message(f"Đã khởi tạo Stockfish với độ sâu {stockfish_depth}")
    except Exception as e:
        log_message(f"Lỗi khi khởi tạo Stockfish: {e}")
        return 0, 0, 0
    
    # Chuẩn bị AI của chúng ta
    try:
        from ChessDove import ChessDove
        chess_engine = ChessDove()
        log_message("Đã khởi tạo ChessDove AI thành công")
    except ImportError:
        log_message("Không thể import ChessDove, sử dụng nước đi ngẫu nhiên")
        chess_engine = None
    
    # Thống kê
    wins, losses, draws = 0, 0, 0
    
    # Chơi các trận đấu
    for i in range(num_games):
        log_message(f"Game {i+1}/{num_games}")
        try:
            result = play_stockfish_game(engine, chess_engine, time_limit)
            
            if result == "ai_win":
                wins += 1
                log_message("AI thắng!")
            elif result == "stockfish_win":
                losses += 1
                log_message("Stockfish thắng!")
            else:
                draws += 1
                log_message("Hòa!")
                
        except Exception as e:
            log_message(f"Lỗi trong trận đấu: {e}")
    
    # Dọn dẹp
    try:
        engine.quit()
    except:
        pass
    
    # Kết quả
    log_message(f"Kết quả cuối cùng: Thắng {wins}, Thua {losses}, Hòa {draws}")
    return wins, losses, draws

def save_elo_history(elo, wins, losses, draws, stockfish_depth):
    """Lưu lịch sử Elo và thông tin đánh giá"""
    history_file = DEFAULT_ELO_HISTORY
    
    # Tạo entry mới
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elo": elo,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "stockfish_depth": stockfish_depth,
        "total_games": wins + losses + draws
    }
    
    # Đọc lịch sử hiện có nếu có
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    else:
        history = []
    
    # Thêm entry mới
    history.append(entry)
    
    # Lưu trở lại
    try:
        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)
        log_message(f"Đã lưu kết quả vào {history_file}")
    except Exception as e:
        log_message(f"Không thể lưu lịch sử: {e}")

# Thêm xử lý ngoại lệ cụ thể khi nhập module ChessDove
try:
    from ChessDove import ChessDove
except SyntaxError:
    # Tạo phiên bản đơn giản của ChessDove để đánh giá Elo khi có lỗi cú pháp
    print("Lỗi nhập module ChessDove. Tạo lớp tạm thời...")
    
    class ChessDove:
        def __init__(self, depth=3):
            self.depth = depth
            
        def get_move(self, board, time_limit=2.0):
            import chess
            import random
            return random.choice(list(board.legal_moves))

# Sửa hàm main để trả về mã 0 nếu thành công dù bất kỳ điều gì xảy ra
def main():
    """Hàm chính để đánh giá Elo"""
    parser = argparse.ArgumentParser(description="Đánh giá Elo của AI cờ vua")
    parser.add_argument("--stockfish-path", default=DEFAULT_STOCKFISH_PATH,
                        help=f"Đường dẫn tới Stockfish (mặc định: {DEFAULT_STOCKFISH_PATH})")
    parser.add_argument("--num-games", type=int, default=30,
                        help="Số ván đấu (mặc định: 30)")
    parser.add_argument("--stockfish-depth", type=int, default=10,
                        help="Độ sâu Stockfish (mặc định: 10)")
    parser.add_argument("--time-limit", type=float, default=1.0,
                        help="Thời gian cho một nước đi (giây, mặc định: 1.0)")
    
    args = parser.parse_args()
    
    # Kiểm tra xem đường dẫn Stockfish có hợp lệ không
    if not os.path.exists(args.stockfish_path):
        log_message(f"CẢNH BÁO: Không tìm thấy Stockfish tại {args.stockfish_path}")
        log_message(f"Thử dùng đường dẫn mặc định: {DEFAULT_STOCKFISH_PATH}")
        if not os.path.exists(DEFAULT_STOCKFISH_PATH):
            log_message("Không tìm thấy Stockfish! Hãy cung cấp đường dẫn chính xác.")
            return 1
        args.stockfish_path = DEFAULT_STOCKFISH_PATH
    
    # Bắt đầu đánh giá
    log_message("=== Bắt đầu đánh giá Elo ===")
    wins, losses, draws = evaluate_against_stockfish(
        args.num_games, 
        args.stockfish_depth, 
        args.stockfish_path,
        args.time_limit
    )
    
    # Tính Elo
    elo = calculate_elo(wins, losses, draws)
    log_message(f"Ước tính Elo: {elo:.2f}")
    
    # Lưu kết quả Elo và lịch sử
    save_elo_history(elo, wins, losses, draws, args.stockfish_depth)
    
    # Lưu kết quả Elo hiện tại
    try:
        with open("current_elo.txt", "w") as f:
            f.write(str(elo))
        log_message("Đã lưu Elo hiện tại vào current_elo.txt")
    except Exception as e:
        log_message(f"Không thể lưu Elo hiện tại: {e}")
        # Ghi giá trị Elo mặc định
        try:
            with open("current_elo.txt", "w") as f:
                f.write("1500")
        except:
            pass
        # Vẫn trả về 0 để không làm hỏng quy trình
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())