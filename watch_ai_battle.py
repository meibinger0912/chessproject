import pygame
import chess
import chess.engine
import os
import sys
import time
from datetime import datetime
import threading
import queue

# Import các module của ChessDove
from ChessDove import ChessDove

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_SQUARE = (118, 150, 86)
LIGHT_SQUARE = (238, 238, 210)
HIGHLIGHT = (255, 255, 0, 50)  # Màu highlight nước đi cuối
INFO_BG = (40, 40, 40)
TEXT_COLOR = (220, 220, 220)
MOVE_LIST_BG = (60, 60, 60)

# Khởi tạo pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("ChessDove vs Stockfish - AI Battle Visualizer")

# Font chữ
font = pygame.font.SysFont("Arial", 18)
title_font = pygame.font.SysFont("Arial", 24, bold=True)
move_font = pygame.font.SysFont("Arial", 16)

# Tải hình ảnh quân cờ
piece_images = {}

# === Thay đổi phần tải hình ảnh quân cờ ===

# Sửa phần tải hình ảnh quân cờ
piece_images = {}

def load_piece_images():
    """Tải hình ảnh quân cờ theo cách đơn giản như trong gui.py"""
    global piece_images
    
    # Sử dụng đường dẫn đến thư mục assets
    asset_path = "D:\\Trí tuệ nhân tạo\\chess project\\assets"
    
    # Tải các quân cờ với cấu trúc tên giống gui.py
    piece_mapping = {
        'P': "white_pawn.png",
        'R': "white_rook.png",
        'N': "white_knight.png",
        'B': "white_bishop.png",
        'Q': "white_queen.png",
        'K': "white_king.png",
        'p': "black_pawn.png",
        'r': "black_rook.png",
        'n': "black_knight.png",
        'b': "black_bishop.png",
        'q': "black_queen.png",
        'k': "black_king.png",
    }
    
    # Tải từng hình ảnh
    for piece_symbol, filename in piece_mapping.items():
        try:
            filepath = os.path.join(asset_path, filename)
            piece_images[piece_symbol] = pygame.transform.scale(
                pygame.image.load(filepath),
                (SQUARE_SIZE, SQUARE_SIZE)
            )
            print(f"Loaded {filename} successfully")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            # Tạo hình mặc định nếu không tìm thấy file
            piece_images[piece_symbol] = create_default_piece(piece_symbol)

def create_default_piece(piece_symbol):
    """Tạo hình mặc định cho quân cờ nếu không tìm thấy file hình ảnh"""
    surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    is_white = piece_symbol.isupper()
    color = WHITE if is_white else BLACK
    
    # Vẽ hình đơn giản cho quân cờ
    pygame.draw.circle(surface, color, (SQUARE_SIZE // 2, SQUARE_SIZE // 2), SQUARE_SIZE // 3)
    
    # Hiển thị ký hiệu của quân cờ
    text = pygame.font.SysFont("Arial", 30, bold=True).render(piece_symbol, True, BLACK if is_white else WHITE)
    text_rect = text.get_rect(center=(SQUARE_SIZE // 2, SQUARE_SIZE // 2))
    surface.blit(text, text_rect)
    return surface
GAME_MODES = ["Normal", "Fast", "Auto"]
class ChessMatch:
    def __init__(self, stockfish_path, chessdove_depth=3, stockfish_depth=10, auto_mode=False):
        self.board = chess.Board()
        self.move_history = []
        self.evaluation_history = []
        self.last_move = None
        self.running = False
        self.paused = False
        self.thinking = False
        self.game_over = False
        self.result = None
        
        # Thông tin trận đấu
        self.move_times = {"chessdove": [], "stockfish": []}
        self.start_time = None
        
        # Khởi tạo ChessDove
        self.chessdove = ChessDove(depth=chessdove_depth)
        
        # Khởi tạo Stockfish
        try:
            self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.stockfish.configure({"Skill Level": stockfish_depth})
            print(f"Initialized Stockfish with depth {stockfish_depth}")
        except Exception as e:
            print(f"Error initializing Stockfish: {e}")
            self.stockfish = None
        
        # Message queue cho communication giữa các thread
        self.message_queue = queue.Queue()
        
        # Thêm thống kê thắng/thua/hòa
        self.stats = {"wins": 0, "losses": 0, "draws": 0}
        
  # Thêm chế độ chơi
        self.mode = "Normal"  # Chế độ mặc định
        self.auto_mode = auto_mode
        self.target_games = 30  # Số trận mặc định

        self.wait_times = {
            "Normal": 0.5,
            "Fast": 0.1,
            "Auto": 0.05
        }

     # Thêm phương thức chuyển đổi chế độ
    def toggle_mode(self):
        """Chuyển đổi giữa các chế độ chơi"""
        mode_index = GAME_MODES.index(self.mode)
        next_mode = GAME_MODES[(mode_index + 1) % len(GAME_MODES)]
        self.mode = next_mode
        
        # Đặt auto_mode dựa trên chế độ hiện tại
        self.auto_mode = (self.mode == "Auto")
        
        # Khởi động lại nếu chuyển sang chế độ Auto và chưa chạy
        if self.mode == "Auto" and not self.running:
            self.start_auto_evaluation(self.target_games)
        
        print(f"Changed to {self.mode} mode")
        return self.mode

    def start(self):
        """Bắt đầu trận đấu"""
        if self.running:
            return
        
        self.running = True
        self.paused = False
        self.game_over = False
        self.board = chess.Board()
        self.move_history = []
        self.evaluation_history = []
        self.move_times = {"chessdove": [], "stockfish": []}
        self.start_time = time.time()
        
        # Bắt đầu thread mới cho AI suy nghĩ
        threading.Thread(target=self.play_game, daemon=True).start()
    
    def pause_resume(self):
        """Tạm dừng hoặc tiếp tục trận đấu"""
        self.paused = not self.paused
    
    def stop(self):
        """Dừng trận đấu"""
        self.running = False
    
    def start_auto_evaluation(self, num_games=30):
        """Bắt đầu chạy nhiều trận đấu tự động để đánh giá Elo"""
        self.target_games = num_games
        self.stats = {"wins": 0, "losses": 0, "draws": 0}
        self.auto_mode = True
        self.start()  # Bắt đầu trận đầu tiên
    
    def play_game(self):
        """Chạy trận đấu giữa ChessDove và Stockfish"""
        while self.running and not self.game_over:
            if self.paused:
                time.sleep(0.1)
                continue
            
            if self.board.is_game_over():
                self.game_over = True
                self.result = self.board.result()
                print(f"Game over: {self.result}")
                
                # Cập nhật thống kê
                if self.result == "1-0":  # ChessDove thắng
                    self.stats["wins"] += 1
                elif self.result == "0-1":  # Stockfish thắng
                    self.stats["losses"] += 1
                else:  # Hòa
                    self.stats["draws"] += 1
                
                # Tự động bắt đầu trận mới trong chế độ đánh giá Elo
                total_games = self.stats["wins"] + self.stats["losses"] + self.stats["draws"]
                if self.auto_mode and total_games < self.target_games:
                    # Tự động bắt đầu trận tiếp theo sau 1 giây
                    time.sleep(1)
                    self.board = chess.Board()
                    self.game_over = False
                    self.move_history = []
                    self.last_move = None
                    continue
                elif self.auto_mode and total_games >= self.target_games:
                    # Đã đủ số trận, tính điểm Elo
                    self.calculate_elo()
                
                break
            
            # Lượt của ChessDove (trắng)
            if self.board.turn == chess.WHITE:
                self.thinking = True
                start_time = time.time()
                
                try:
                    move = self.chessdove.get_move(self.board, time_limit=2.0)
                    if move:
                        self.board.push(move)
                        self.last_move = move
                        self.move_history.append(move)
                        
                        # Lưu thời gian suy nghĩ
                        elapsed = time.time() - start_time
                        self.move_times["chessdove"].append(elapsed)
                        
                        # Đánh giá vị trí
                        try:
                            evaluation = self.evaluate_position()
                            self.evaluation_history.append(evaluation)
                        except:
                            self.evaluation_history.append(0)
                    
                    else:
                        print("ChessDove couldn't find a valid move")
                        self.game_over = True
                
                except Exception as e:
                    print(f"Error during ChessDove's turn: {e}")
                    self.game_over = True
                
                self.thinking = False
            
            # Lượt của Stockfish (đen)
            else:
                if self.stockfish:
                    self.thinking = True
                    start_time = time.time()
                    
                    try:
                        result = self.stockfish.play(
                            self.board,
                            chess.engine.Limit(time=1.0)
                        )
                        self.board.push(result.move)
                        self.last_move = result.move
                        self.move_history.append(result.move)
                        
                        # Lưu thời gian suy nghĩ
                        elapsed = time.time() - start_time
                        self.move_times["stockfish"].append(elapsed)
                        
                        # Đánh giá vị trí
                        try:
                            evaluation = self.evaluate_position()
                            self.evaluation_history.append(evaluation)
                        except:
                            self.evaluation_history.append(0)
                    
                    except Exception as e:
                        print(f"Error during Stockfish's turn: {e}")
                        self.game_over = True
                    
                    self.thinking = False
            
            # Thông báo cho giao diện cập nhật
            self.message_queue.put("update")
            
            # Tạm dừng để hiển thị - giảm thời gian nếu ở chế độ tự động
            if self.auto_mode:
                time.sleep(0.1)  # Giảm thời gian chờ trong chế độ tự động
            else:
               wait_time = self.wait_times.get(self.mode, 0.5)
               time.sleep(wait_time)
    
    def evaluate_position(self):
        """Đánh giá vị trí bàn cờ hiện tại"""
        # Sử dụng đánh giá của ChessDove
        return self.chessdove.evaluate_position(self.board)
    
    def get_game_info(self):
        """Lấy thông tin trận đấu"""
        # Tính thời gian trung bình mỗi nước đi
        avg_time_chessdove = sum(self.move_times["chessdove"]) / len(self.move_times["chessdove"]) if self.move_times["chessdove"] else 0
        avg_time_stockfish = sum(self.move_times["stockfish"]) / len(self.move_times["stockfish"]) if self.move_times["stockfish"] else 0
        
        # Đếm số nước đi của mỗi bên
        chessdove_moves = len(self.move_times["chessdove"])
        stockfish_moves = len(self.move_times["stockfish"])
        
        # Thời gian trận đấu
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Đánh giá vị trí hiện tại
        current_evaluation = self.evaluation_history[-1] if self.evaluation_history else 0
        
        return {
            "chessdove_moves": chessdove_moves,
            "stockfish_moves": stockfish_moves,
            "avg_time_chessdove": avg_time_chessdove,
            "avg_time_stockfish": avg_time_stockfish,
            "elapsed": elapsed,
            "evaluation": current_evaluation,
            "wins": self.stats["wins"],
            "losses": self.stats["losses"],
            "draws": self.stats["draws"],
            "status": "Thinking" if self.thinking else "Paused" if self.paused else "Running",
            "result": self.result if self.game_over else None
        }
    
    def calculate_elo(self):
        """Tính điểm Elo dựa trên kết quả đấu với Stockfish"""
        # Công thức Elo cơ bản
        total_games = self.stats["wins"] + self.stats["losses"] + self.stats["draws"] 
        if total_games == 0:
            return
            
        # Điểm hiệu suất
        performance = (self.stats["wins"] + self.stats["draws"] * 0.5) / total_games
        
        # Stockfish có Elo xác định (khoảng 2200 với độ sâu 10)
        stockfish_elo = 2200  # Có thể điều chỉnh dựa theo độ sâu của Stockfish
        
        # Công thức Elo
        if performance == 1:
            elo_diff = 800  # Thắng tất cả
        elif performance == 0:
            elo_diff = -800  # Thua tất cả
        else:
            elo_diff = -400 * math.log10(1/performance - 1)
            
        chessdove_elo = stockfish_elo + elo_diff
        
        # Hiển thị và lưu kết quả
        print(f"\n=== ELO EVALUATION RESULTS ===")
        print(f"Games played: {total_games}")
        print(f"Wins: {self.stats['wins']} | Draws: {self.stats['draws']} | Losses: {self.stats['losses']}")
        print(f"Performance: {performance:.2f}")
        print(f"ChessDove Elo rating: {chessdove_elo:.0f}")
        
        # Lưu vào file
        try:
            with open("current_elo.txt", "w") as f:
                f.write(f"{chessdove_elo:.0f}")
        except Exception as e:
            print(f"Error saving Elo: {e}")
            
        self.elo = chessdove_elo
    
    def cleanup(self):
        """Giải phóng tài nguyên"""
        if self.stockfish:
            self.stockfish.quit()

def draw_board(screen, board, last_move=None):
    """Vẽ bàn cờ với các quân cờ"""
    # Vẽ các ô cờ
    for row in range(8):
        for col in range(8):
            square = row * 8 + col
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    
    # Highlight nước đi cuối cùng
    if last_move:
        from_sq = last_move.from_square
        to_sq = last_move.to_square
        
        for sq in [from_sq, to_sq]:
            col = chess.square_file(sq)
            row = chess.square_rank(sq)
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_surface.fill(HIGHLIGHT)
            screen.blit(highlight_surface, (col * SQUARE_SIZE, (7 - row) * SQUARE_SIZE))
    
    # Vẽ các quân cờ
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            col = chess.square_file(square)
            row = chess.square_rank(square)
            piece_symbol = piece.symbol()
            
            if piece_symbol in piece_images:
                screen.blit(piece_images[piece_symbol], 
                           (col * SQUARE_SIZE, (7 - row) * SQUARE_SIZE))

def draw_move_list(screen, moves, x, y, width, height):
    """Vẽ danh sách nước đi"""
    # Vẽ nền
    pygame.draw.rect(screen, MOVE_LIST_BG, pygame.Rect(x, y, width, height))
    
    # Tiêu đề
    title_text = title_font.render("Move History", True, TEXT_COLOR)
    screen.blit(title_text, (x + 10, y + 10))
    
    # Danh sách nước đi
    move_y = y + 50
    moves_per_row = 2
    
    for i in range(0, len(moves), 2):
        # Số thứ tự nước đi
        move_number = i // 2 + 1
        number_text = move_font.render(f"{move_number}.", True, TEXT_COLOR)
        screen.blit(number_text, (x + 10, move_y))
        
        # Nước đi của trắng
        if i < len(moves):
            white_text = move_font.render(moves[i].uci(), True, WHITE)
            screen.blit(white_text, (x + 40, move_y))
        
        # Nước đi của đen
        if i + 1 < len(moves):
            black_text = move_font.render(moves[i + 1].uci(), True, TEXT_COLOR)
            screen.blit(black_text, (x + 100, move_y))
        
        # Xuống dòng sau mỗi cặp nước đi
        move_y += 25
        
        # Giới hạn số dòng hiển thị
        if move_y > y + height - 30:
            break

def draw_info_panel(screen, match_info, x, y, width, height):
    """Vẽ bảng thông tin trận đấu"""
    # Vẽ nền
    pygame.draw.rect(screen, INFO_BG, pygame.Rect(x, y, width, height))
    
    # Tiêu đề
    title_text = title_font.render("Game Information", True, TEXT_COLOR)
    screen.blit(title_text, (x + 10, y + 10))
    
    # Thông tin
    info_y = y + 50
    info_texts = [
        f"Moves: ChessDove: {match_info['chessdove_moves']} | Stockfish: {match_info['stockfish_moves']}",
        f"Avg time/move: ChessDove: {match_info['avg_time_chessdove']:.2f}s | Stockfish: {match_info['avg_time_stockfish']:.2f}s",
        f"Game duration: {int(match_info['elapsed'] // 60)}:{int(match_info['elapsed'] % 60):02d}",
        f"Current evaluation: {match_info['evaluation']:.2f}",
        f"Status: {match_info['status']}",
        f"Wins: {match_info['wins']} | Losses: {match_info['losses']} | Draws: {match_info['draws']}",
    ]
    
    if match_info['result']:
        info_texts.append(f"Result: {match_info['result']}")
    
    for text in info_texts:
        text_surface = font.render(text, True, TEXT_COLOR)
        screen.blit(text_surface, (x + 10, info_y))
        info_y += 30

def draw_evaluation_graph(screen, evaluation_history, x, y, width, height):
    """Vẽ đồ thị đánh giá"""
    # Vẽ nền
    pygame.draw.rect(screen, INFO_BG, pygame.Rect(x, y, width, height))
    
    # Tiêu đề
    title_text = title_font.render("Evaluation Graph", True, TEXT_COLOR)
    screen.blit(title_text, (x + 10, y + 10))
    
    # Vẽ đường cơ sở (đánh giá = 0)
    pygame.draw.line(screen, (100, 100, 100), (x + 10, y + height // 2), (x + width - 10, y + height // 2), 1)
    
    # Nếu không có dữ liệu đánh giá
    if not evaluation_history:
        no_data_text = font.render("No evaluation data yet", True, TEXT_COLOR)
        screen.blit(no_data_text, (x + width // 2 - 80, y + height // 2 - 10))
        return
    
    # Chuẩn hóa đánh giá để vẽ đồ thị
    max_eval = max(1.0, max([abs(e) for e in evaluation_history]))
    scale_factor = (height - 60) / (2 * max_eval)
    
    # Vẽ đường đánh giá
    points = []
    for i, eval_value in enumerate(evaluation_history):
        # Giới hạn giá trị đánh giá để đồ thị không bị quá lớn
        if abs(eval_value) > 10:
            eval_value = 10 * (1 if eval_value > 0 else -1)
            
        point_x = x + 20 + (width - 40) * i / max(1, len(evaluation_history) - 1)
        point_y = y + height // 2 - eval_value * scale_factor
        points.append((point_x, point_y))
    
    if len(points) > 1:
        pygame.draw.lines(screen, (0, 200, 0), False, points, 2)

# Sửa định nghĩa hàm để khớp với cách gọi
def draw_control_panel(screen, match, x, y, width, height):
    """Vẽ bảng điều khiển"""
    # Vẽ nền
    pygame.draw.rect(screen, INFO_BG, pygame.Rect(x, y, width, height))
    
    # Các nút điều khiển
    button_width = 120
    button_height = 40
    button_margin = 20
    
    # Nút Start/Stop
    start_stop_color = (200, 50, 50) if match.running else (50, 200, 50)
    start_stop_text = "Stop" if match.running else "Start"
    
    pygame.draw.rect(screen, start_stop_color, 
                    pygame.Rect(x + button_margin, y + button_margin, button_width, button_height))
    
    text = font.render(start_stop_text, True, WHITE)
    screen.blit(text, (x + button_margin + button_width // 2 - text.get_width() // 2, 
                      y + button_margin + button_height // 2 - text.get_height() // 2))
    
    # Nút Pause/Resume
    if match.running:
        pause_resume_color = (200, 200, 50) if match.paused else (50, 50, 200)
        pause_resume_text = "Resume" if match.paused else "Pause"
        
        pygame.draw.rect(screen, pause_resume_color, 
                        pygame.Rect(x + 2 * button_margin + button_width, y + button_margin, button_width, button_height))
        
        text = font.render(pause_resume_text, True, WHITE)
        screen.blit(text, (x + 2 * button_margin + button_width * 1.5 - text.get_width() // 2, 
                          y + button_margin + button_height // 2 - text.get_height() // 2))
    
    # Nút Mode
    mode_color = {
        "Normal": (100, 100, 200),
        "Fast": (100, 200, 100),
        "Auto": (200, 100, 100)
    }.get(match.mode, (150, 150, 150))
    
    pygame.draw.rect(screen, mode_color, 
                    pygame.Rect(x + 3 * button_margin + 2 * button_width, y + button_margin, button_width, button_height))
    
    text = font.render(f"Mode: {match.mode}", True, WHITE)
    screen.blit(text, (x + 3 * button_margin + 2 * button_width + button_width // 2 - text.get_width() // 2, 
                      y + button_margin + button_height // 2 - text.get_height() // 2))
def draw_stats_panel(screen, match_info, x, y, width, height):
    """Vẽ bảng thống kê thắng/thua/hòa"""
    # Vẽ nền
    pygame.draw.rect(screen, INFO_BG, pygame.Rect(x, y, width, height))
    
    # Tiêu đề
    title_text = title_font.render("Match Statistics", True, TEXT_COLOR)
    screen.blit(title_text, (x + 10, y + 10))
    
    # Vẽ thống kê
    total_games = match_info["wins"] + match_info["losses"] + match_info["draws"]
    
    # Vẽ thông tin số trận
    stats_y = y + 50
    
    # Sửa phần này để tránh lỗi nếu total_games = 0
    if total_games > 0:
        win_percent = match_info['wins']*100/total_games
        loss_percent = match_info['losses']*100/total_games
        draw_percent = match_info['draws']*100/total_games
    else:
        win_percent = loss_percent = draw_percent = 0.0
    
    stats_texts = [
        f"Total Games: {total_games}",
        f"ChessDove Wins: {match_info['wins']} ({win_percent:.1f}%)",
        f"Stockfish Wins: {match_info['losses']} ({loss_percent:.1f}%)",
        f"Draws: {match_info['draws']} ({draw_percent:.1f}%)",
    ]
    
    for text in stats_texts:
        text_surface = font.render(text, True, TEXT_COLOR)
        screen.blit(text_surface, (x + 10, stats_y))
        stats_y += 30
    
    # Vẽ biểu đồ thanh ngang
    if total_games > 0:
        bar_y = stats_y + 20
        bar_height = 30
        bar_width = width - 40
        
        # Vẽ border
        pygame.draw.rect(screen, WHITE, pygame.Rect(x + 20, bar_y, bar_width, bar_height), 1)
        
        # Vẽ các phần của biểu đồ
        win_width = int(bar_width * match_info["wins"] / total_games)
        draw_width = int(bar_width * match_info["draws"] / total_games)
        loss_width = bar_width - win_width - draw_width
        
        # Màu sắc
        win_color = (50, 200, 50)    # Xanh lá
        draw_color = (200, 200, 50)  # Vàng
        loss_color = (200, 50, 50)   # Đỏ
        
        # Vẽ phần thắng
        if win_width > 0:
            pygame.draw.rect(screen, win_color, pygame.Rect(x + 20, bar_y, win_width, bar_height))
        
        # Vẽ phần hòa
        if draw_width > 0:
            pygame.draw.rect(screen, draw_color, pygame.Rect(x + 20 + win_width, bar_y, draw_width, bar_height))
        
        # Vẽ phần thua
        if loss_width > 0:
            pygame.draw.rect(screen, loss_color, pygame.Rect(x + 20 + win_width + draw_width, bar_y, loss_width, bar_height))

def main(stockfish_path="D:\\Trí tuệ nhân tạo\\chess project\\stockfish\\stockfish.exe", 
         chessdove_depth=5, stockfish_depth=10, auto_mode=False, num_games=30):
    """Hàm chính chạy game"""
    
    # Tải hình ảnh quân cờ
    load_piece_images()
    
    # Tạo trận đấu
    match = ChessMatch(stockfish_path, chessdove_depth, stockfish_depth, auto_mode)
    
    # Nếu ở chế độ tự động, bắt đầu đánh giá Elo
    if auto_mode:
        match.start_auto_evaluation(num_games)
    
    # Vòng lặp chính
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Xử lý click chuột
                mouse_pos = pygame.mouse.get_pos()
                
                # Kiểm tra click vào nút Start/Stop
                if (BOARD_SIZE + 20 <= mouse_pos[0] <= BOARD_SIZE + 20 + 120 and
                    BOARD_SIZE - 70 <= mouse_pos[1] <= BOARD_SIZE - 70 + 40):
                    if match.running:
                        match.stop()
                    else:
                        match.start()
                
                # Kiểm tra click vào nút Pause/Resume
                elif (BOARD_SIZE + 2 * 20 + 120 <= mouse_pos[0] <= BOARD_SIZE + 2 * 20 + 2 * 120 and
                      BOARD_SIZE - 70 <= mouse_pos[1] <= BOARD_SIZE - 70 + 40):
                    if match.running:
                        match.pause_resume()
                
                # Kiểm tra click vào nút Mode
                elif (BOARD_SIZE + 3 * 20 + 2 * 120 <= mouse_pos[0] <= BOARD_SIZE + 3 * 20 + 3 * 120 and
                      BOARD_SIZE - 70 <= mouse_pos[1] <= BOARD_SIZE - 70 + 40):
                    match.toggle_mode()
                    # Tạo file hướng dẫn nếu chưa tồn tại
                   
        
        # Kiểm tra message từ thread đấu trận
        try:
            while not match.message_queue.empty():
                message = match.message_queue.get_nowait()
                if message == "update":
                    pass  # Chúng ta sẽ cập nhật giao diện trong mỗi frame
        except queue.Empty:
            pass
        
        # Vẽ giao diện
        screen.fill(BLACK)
        
        # Vẽ bàn cờ
        draw_board(screen, match.board, match.last_move)
        
        # Vẽ danh sách nước đi
        draw_move_list(screen, match.move_history, BOARD_SIZE + 20, 20, SCREEN_WIDTH - BOARD_SIZE - 40, 350)
        
        # Vẽ thông tin trận đấu
        match_info = match.get_game_info()
        draw_info_panel(screen, match_info, 20, BOARD_SIZE + 20, BOARD_SIZE - 40, 140)
        
        # Vẽ bảng thống kê thay vì đồ thị đánh giá
        draw_stats_panel(screen, match_info, BOARD_SIZE + 20, 390, 
                        SCREEN_WIDTH - BOARD_SIZE - 40, 300)
        
        # Vẽ bảng điều khiển
        draw_control_panel(screen, match, 
                          BOARD_SIZE + 20, BOARD_SIZE - 70, 
                          SCREEN_WIDTH - BOARD_SIZE - 40, 80)
        
        # Cập nhật màn hình
        pygame.display.flip()
        clock.tick(30)  # 30 FPS
    
    # Giải phóng tài nguyên
    match.cleanup()
    pygame.quit()

def update_elo_rating(result, opponent_elo=1500):
    """
    Cập nhật Elo dựa trên kết quả trận đấu
    result: 1 cho thắng, 0.5 cho hòa, 0 cho thua
    """
    elo_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_elo.txt")
    
    # Đọc Elo hiện tại
    try:
        with open(elo_file_path, "r") as f:
            current_elo = float(f.read().strip())
    except:
        current_elo = 1500  # Giá trị mặc định
    
    # Hằng số K (ảnh hưởng của một trận đấu)
    k = 32  # Giá trị K tiêu chuẩn trong xếp hạng Elo
    
    # Tính điểm kỳ vọng
    expected = 1 / (1 + 10 ** ((opponent_elo - current_elo) / 400))
    
    # Cập nhật Elo
    new_elo = current_elo + k * (result - expected)
    
    # Làm tròn và lưu Elo mới
    new_elo = round(new_elo)
    
    with open(elo_file_path, "w") as f:
        f.write(str(new_elo))
    
    print(f"Elo updated: {current_elo:.1f} -> {new_elo:.1f} (match result: {result})")
    return new_elo

if __name__ == "__main__":
    # Xử lý tham số dòng lệnh
    import argparse
    import math
    import subprocess
    
    parser = argparse.ArgumentParser(description="Chess battle visualization between ChessDove and Stockfish")
    parser.add_argument("--stockfish-path", type=str, 
                        default="D:\\Trí tuệ nhân tạo\\chess project\\stockfish\\stockfish.exe",
                        help="Path to stockfish.exe")
    parser.add_argument("--chessdove-depth", type=int, default=3,
                        help="ChessDove search depth")
    parser.add_argument("--stockfish-depth", type=int, default=5,
                        help="Stockfish strength level")
    parser.add_argument("--auto-mode", action="store_true", 
                        help="Run in automatic mode for Elo evaluation")
    parser.add_argument("--num-games", type=int, default=30,
                        help="Number of games to play for Elo evaluation")
    
    args = parser.parse_args()
    
    # Chạy chương trình chính với các tham số
    main(args.stockfish_path, args.chessdove_depth, args.stockfish_depth, 
         args.auto_mode, args.num_games)
    
    # Thay
    # subprocess.Popen(["python", "D:/Trí tuệ nhân tạo/chess project/ChessDove/watch_ai_battle.py"])
    
    # Bằng
    if getattr(sys, 'frozen', False):
        script_path = os.path.join(sys._MEIPASS, "ChessDove", "watch_ai_battle.py")
    else:
        script_path = os.path.join(os.path.dirname(__file__), "ChessDove", "watch_ai_battle.py")
    subprocess.Popen([sys.executable, script_path])