import pygame
import chess
import time
import sys
import os
from graphic import (
    BOARD_SIZE, SCORE_PANEL_WIDTH, WIDTH, HEIGHT, SQUARE_SIZE,
    WHITE, BLACK, GRAY, RED, GREEN,
    load_images, should_flip_board, draw_board_base, 
    draw_pieces, draw_highlights, draw_end_game_message, draw_button, draw_menu
)

# Khởi tạo pygame
pygame.init()

def draw_chessdove_info(screen):
    """Hiển thị thông tin về ChessDove AI."""
    screen.fill(WHITE)
    
    # Đọc Elo đã train được từ file
    try:
        with open("D:/Trí tuệ nhân tạo/chess project/ChessDove/current_elo.txt", "r") as f:
            current_elo = f.read().strip()
    except:
        current_elo = "1500"  # Giá trị mặc định nếu không đọc được file

    # Tiêu đề
    title_font = pygame.font.SysFont("Arial", 48, bold=True)
    title_text = title_font.render("ChessDove AI", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 80))
    
    # Thông tin
    info_font = pygame.font.SysFont("Arial", 36)
    elo_text = info_font.render(f"Current Elo: {current_elo}", True, BLACK)
    screen.blit(elo_text, (WIDTH // 2 - elo_text.get_width() // 2, 180))
    
    # Thông tin bổ sung
    info_items = [
        "Self-trained Chess AI",
        "Based on neural network evaluation",
        "Trained through self-play and Stockfish analysis"
    ]
    
    desc_font = pygame.font.SysFont("Arial", 28)
    for i, text in enumerate(info_items):
        info_surface = desc_font.render(text, True, BLACK)
        y_pos = 260 + i * 50
        screen.blit(info_surface, (WIDTH // 2 - info_surface.get_width() // 2, y_pos))

    # Tiêu đề chọn màu
    side_title = info_font.render("Choose Your Side:", True, BLACK)
    screen.blit(side_title, (WIDTH // 2 - side_title.get_width() // 2, 450))
    
    # Nút chọn màu
    buttons = []
    
    # Nút trắng
    white_rect = pygame.Rect(WIDTH // 2 - 200, 520, 180, 80)
    pygame.draw.rect(screen, (240, 240, 240), white_rect)
    pygame.draw.rect(screen, BLACK, white_rect, 3)
    white_text = info_font.render("White", True, BLACK)
    screen.blit(white_text, (white_rect.x + white_rect.width // 2 - white_text.get_width() // 2,
                            white_rect.y + white_rect.height // 2 - white_text.get_height() // 2))
    buttons.append((white_rect, "white"))
    
    # Nút đen
    black_rect = pygame.Rect(WIDTH // 2 + 20, 520, 180, 80)
    pygame.draw.rect(screen, (50, 50, 50), black_rect)
    pygame.draw.rect(screen, WHITE, black_rect, 3)
    black_text = info_font.render("Black", True, WHITE)
    screen.blit(black_text, (black_rect.x + black_rect.width // 2 - black_text.get_width() // 2,
                            black_rect.y + black_rect.height // 2 - black_text.get_height() // 2))
    buttons.append((black_rect, "black"))
    
    # Nút Back
    back_rect = pygame.Rect(20, HEIGHT - 100, 120, 60)
    pygame.draw.rect(screen, GRAY, back_rect)
    pygame.draw.rect(screen, (100, 100, 100), back_rect, 3)
    back_text = pygame.font.SysFont("Arial", 32).render("Back", True, BLACK)
    screen.blit(back_text, (back_rect.x + back_rect.width // 2 - back_text.get_width() // 2,
                           back_rect.y + back_rect.height // 2 - back_text.get_height() // 2))
    buttons.append((back_rect, "back"))
    
    return buttons

# Sửa hàm draw_chessdove_panel()
def draw_chessdove_panel(screen, game):
    """Vẽ bảng thông tin điểm và đồng hồ cho ChessDove"""
    score_panel = pygame.Rect(0, 0, SCORE_PANEL_WIDTH, HEIGHT)
    pygame.draw.rect(screen, (240, 240, 240), score_panel)
    pygame.draw.line(screen, (200, 200, 200), (SCORE_PANEL_WIDTH, 0), 
                    (SCORE_PANEL_WIDTH, HEIGHT), 3)

    # Tạo fonts
    title_font = pygame.font.SysFont("Arial", 48)
    text_font = pygame.font.SysFont("Arial", 36)
    timer_font = pygame.font.SysFont("Arial", 42)

    # Vẽ tiêu đề
    title = title_font.render("CHESSDOVE", True, BLACK)
    title_rect = title.get_rect(center=(SCORE_PANEL_WIDTH // 2, 80))
    screen.blit(title, title_rect)
    
    # Đọc Elo đã train được từ file
    try:
        with open("D:/Trí tuệ nhân tạo/chess project/ChessDove/current_elo.txt", "r") as f:
            current_elo = f.read().strip()
    except:
        current_elo = "1500"  # Giá trị mặc định nếu không đọc được file
    
    # Vẽ Elo
    elo_text = text_font.render(f"Elo: {current_elo}", True, BLACK)
    elo_rect = elo_text.get_rect(center=(SCORE_PANEL_WIDTH // 2, 140))
    screen.blit(elo_text, elo_rect)

    # Vẽ điểm số
    white_text = text_font.render(f"White: {game.white_score}", True, BLACK)
    black_text = text_font.render(f"Black: {game.black_score}", True, BLACK)
    screen.blit(white_text, (20, 200))
    screen.blit(black_text, (20, 250))

    # Vẽ đồng hồ
    if hasattr(game, 'clock'):
        # Cập nhật đồng hồ và kiểm tra thời gian
        if not game.clock.update() and not game.board.is_game_over():
            game.game_result = "Time's up! Move took too long."
            return

        # Thời gian chơi
        game_time = text_font.render("Game Time:", True, BLACK)
        game_time_val = timer_font.render(
            game.clock.format_time(game.clock.game_time), 
            True, 
            BLACK
        )
        screen.blit(game_time, (20, 320))
        screen.blit(game_time_val, (20, 360))

        # Bộ đếm thời gian nước đi
        move_time = text_font.render("Move Time:", True, BLACK)
        move_color = RED if game.clock.move_time > 8 else BLACK
        move_time_val = timer_font.render(
            f"{game.clock.move_time:.1f}s", 
            True, 
            move_color
        )
        screen.blit(move_time, (20, 420))
        screen.blit(move_time_val, (20, 460))

    # Status - depth search 
    # Sửa dòng này:
    if hasattr(game, 'chessdove_ai') and game.chessdove_ai:
        # Kiểm tra xem có thuộc tính depth không
        depth_value = "Unknown"
        if hasattr(game.chessdove_ai, 'depth'):
            depth_value = str(game.chessdove_ai.depth)
        elif hasattr(game.chessdove_ai, 'get_depth'):
            depth_value = str(game.chessdove_ai.get_depth())
        
        depth_text = text_font.render(f"Depth: {depth_value}", True, BLACK)
        screen.blit(depth_text, (20, 520))

    # Vẽ nút back
    button_rect = pygame.Rect(20, HEIGHT - 100, SCORE_PANEL_WIDTH - 40, 60)
    return draw_button(screen, button_rect, "Back to Menu")

def initialize_chessdove(stockfish_path):
    """Khởi tạo ChessDove AI"""
    import sys
    import os
    
    # QUAN TRỌNG: Thêm đường dẫn vào sys.path trước khi import
    chessdove_dir = os.path.abspath("D:/Trí tuệ nhân tạo/chess project/ChessDove")
    if chessdove_dir not in sys.path:
        sys.path.insert(0, chessdove_dir)
    
    try:
        from ChessDove import ChessDove
        
        # Đọc cấu hình ChessDove từ file nếu cần
        chessdove_depth = 3  # Độ sâu mặc định
        try:
            with open(os.path.join(chessdove_dir, "config.txt"), "r") as f:
                for line in f:
                    if line.startswith("depth="):
                        chessdove_depth = int(line.strip().split("=")[1])
        except:
            pass
        
        # Tạo đối tượng ChessDove AI
        return ChessDove(depth=chessdove_depth)
    except Exception as e:
        print(f"Lỗi khi tải ChessDove AI: {e}")
        return None

def run_chessdove_game(player_side, stockfish_path):
    """Chạy trò chơi với ChessDove AI"""
    # Thiết lập màn hình
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    # Sửa lỗi: setCaption -> set_caption
    pygame.display.set_caption('Chess vs ChessDove AI')
    
    # Tải hình ảnh
    board_image, piece_images, highlight_images = load_images()
    
    # Khởi tạo AI
    chessdove_ai = initialize_chessdove(stockfish_path)
    if chessdove_ai is None:
        print("Không thể khởi tạo ChessDove AI, trở về menu chính")
        return
    
    # Khởi tạo game
    from chess_logic import ChessGame
    game = ChessGame(1500, stockfish_path)  # Elo không quan trọng cho ChessDove
    game.chessdove_ai = chessdove_ai
    game.use_chessdove = True
    game.set_player_side(player_side)
    
    # Vòng lặp game
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and game.is_game_over():
                if event.key == pygame.K_SPACE:
                    return  # Trở về menu
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Kiểm tra click vào nút Back
                button_rect = draw_chessdove_panel(screen, game)
                if button_rect and button_rect.collidepoint(mouse_pos):
                    return  # Trở về menu
                
                # Xử lý click trên bàn cờ
                if not game.is_game_over():
                    board_x = mouse_pos[0] - SCORE_PANEL_WIDTH
                    if 0 <= board_x < BOARD_SIZE:
                        col = board_x // SQUARE_SIZE
                        row = 7 - (mouse_pos[1] // SQUARE_SIZE)
                        
                        # Cần đảo nếu người chơi chọn đen
                        if should_flip_board(game.player_side):
                            col = 7 - col
                            row = 7 - row
                            
                        if 0 <= col < 8 and 0 <= row < 8:
                            game.handle_click(row, col)
        
        # Vẽ bàn cờ
        screen.fill(WHITE)
        
        # Vẽ bảng thông tin
        draw_chessdove_panel(screen, game)
        
        # Vẽ bàn cờ nền
        draw_board_base(screen, board_image, SCORE_PANEL_WIDTH)
        
        # Vẽ highlight
        flip = should_flip_board(game.player_side)
        legal_moves = game.get_legal_moves_for_selected_piece()
        draw_highlights(screen, game.board, game.selected_square, legal_moves, 
                       highlight_images, SCORE_PANEL_WIDTH, flip)
        
        # Vẽ quân cờ
        draw_pieces(screen, game.board, piece_images, SCORE_PANEL_WIDTH, flip)
        
        # Hiển thị thông báo kết thúc nếu game over
        if game.is_game_over():
            result = game.get_game_result()
            if result:
                draw_end_game_message(screen, result)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    return