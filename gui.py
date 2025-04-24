import pygame
import chess
import time
import sys
from graphic import (
    BOARD_SIZE, SCORE_PANEL_WIDTH, WIDTH, HEIGHT, SQUARE_SIZE,
    WHITE, BLACK, GRAY, RED, GREEN,
    load_images, should_flip_board, draw_board_base, 
    draw_pieces, draw_highlights, draw_end_game_message, draw_button, draw_menu
)

# Khởi tạo pygame
pygame.init()

def draw_elo_selection(screen, width, height):
    """Vẽ màn hình chọn Elo."""
    # Các mức Elo
    elo_options = [
        ("Elo 1200", 1200),
        ("Elo 2200", 2200),
        ("Elo 2500", 2500),
        ("Elo 3000", 3000)
    ]
    
    # Sử dụng hàm menu chung
    return draw_menu(screen, "Choose Stockfish Elo", elo_options)

def draw_side_selection(screen, width, height):
    """Vẽ màn hình chọn bên."""
    # Các bên
    side_options = [
        ("Play as White", "white"),
        ("Play as Black", "black")
    ]
    
    # Sử dụng hàm menu chung
    return draw_menu(screen, "Choose Your Side", side_options, None)

def choose_side(screen):
    """Hiển thị màn hình chọn bên và trả về lựa chọn của người chơi."""
    while True:
        buttons = draw_side_selection(screen, WIDTH, HEIGHT)
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for button, side in buttons:
                    if button.collidepoint(mouse_pos):
                        return side

def draw_stockfish_panel(screen, game):
    """Vẽ bảng thông tin điểm và đồng hồ cho Stockfish"""
    score_panel = pygame.Rect(0, 0, SCORE_PANEL_WIDTH, HEIGHT)
    pygame.draw.rect(screen, (240, 240, 240), score_panel)
    pygame.draw.line(screen, (200, 200, 200), (SCORE_PANEL_WIDTH, 0), 
                    (SCORE_PANEL_WIDTH, HEIGHT), 3)

    # Tạo fonts
    title_font = pygame.font.SysFont("Arial", 48)
    text_font = pygame.font.SysFont("Arial", 36)
    timer_font = pygame.font.SysFont("Arial", 42)

    # Vẽ tiêu đề
    title = title_font.render("STOCKFISH", True, BLACK)
    title_rect = title.get_rect(center=(SCORE_PANEL_WIDTH // 2, 80))
    screen.blit(title, title_rect)
    
    # Vẽ Elo level
    elo_text = text_font.render(f"Elo: {game.stockfish.get_parameters()['UCI_Elo']}", True, BLACK)
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

        # Thời gian tổng của người chơi
        white_total = text_font.render(
            f"White: {game.clock.format_time(game.clock.white_time)}", 
            True, 
            BLACK
        )
        black_total = text_font.render(
            f"Black: {game.clock.format_time(game.clock.black_time)}", 
            True, 
            BLACK
        )
        screen.blit(white_total, (20, 520))
        screen.blit(black_total, (20, 570))

    # Vẽ nút back
    button_rect = pygame.Rect(20, HEIGHT - 100, SCORE_PANEL_WIDTH - 40, 60)
    return draw_button(screen, button_rect, "Back to Menu")

def run_stockfish_game(elo_rating, player_side, stockfish_path):
    """Chạy trò chơi với Stockfish ở mức Elo được chọn"""
    # Thiết lập màn hình
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Chess vs Stockfish - Elo {elo_rating}')
    
    # Tải hình ảnh
    board_image, piece_images, highlight_images = load_images()
    
    # Khởi tạo game
    from chess_logic import ChessGame
    game = ChessGame(elo_rating, stockfish_path)
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
                button_rect = draw_stockfish_panel(screen, game)
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
        draw_stockfish_panel(screen, game)
        
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