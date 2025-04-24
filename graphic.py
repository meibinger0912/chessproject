import pygame
import chess
import os

# Khởi tạo pygame nếu cần
if not pygame.get_init():
    pygame.init()

# Cấu hình kích thước
BOARD_SIZE = 1024
SCORE_PANEL_WIDTH = 200
WIDTH = SCORE_PANEL_WIDTH + BOARD_SIZE
HEIGHT = BOARD_SIZE
SQUARE_SIZE = BOARD_SIZE // 8  # 128x128 pixels

# Các màu thường dùng
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
LIGHT_BLUE = (173, 216, 230)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Tải hình ảnh
def load_images(assets_path="assets"):
    """Tải hình ảnh quân cờ và bàn cờ"""
    board_image = pygame.image.load(os.path.join(assets_path, "board.png"))
    board_image = pygame.transform.scale(board_image, (BOARD_SIZE, BOARD_SIZE))
    
    piece_images = {
        "P": pygame.image.load(os.path.join(assets_path, "white_pawn.png")),
        "p": pygame.image.load(os.path.join(assets_path, "black_pawn.png")),
        "R": pygame.image.load(os.path.join(assets_path, "white_rook.png")),
        "r": pygame.image.load(os.path.join(assets_path, "black_rook.png")),
        "N": pygame.image.load(os.path.join(assets_path, "white_knight.png")),
        "n": pygame.image.load(os.path.join(assets_path, "black_knight.png")),
        "B": pygame.image.load(os.path.join(assets_path, "white_bishop.png")),
        "b": pygame.image.load(os.path.join(assets_path, "black_bishop.png")),
        "Q": pygame.image.load(os.path.join(assets_path, "white_queen.png")),
        "q": pygame.image.load(os.path.join(assets_path, "black_queen.png")),
        "K": pygame.image.load(os.path.join(assets_path, "white_king.png")),
        "k": pygame.image.load(os.path.join(assets_path, "black_king.png")),
    }
    
    # Điều chỉnh kích thước hình ảnh quân cờ
    for key in piece_images:
        piece_images[key] = pygame.transform.scale(piece_images[key], (SQUARE_SIZE, SQUARE_SIZE))
    
    highlight_images = {
        "selected": pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA),
        "legal": pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA),
        "illegal": pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA),
        "check": pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    }
    
    # Tạo các hiệu ứng highlight
    highlight_images["selected"].fill((255, 255, 0, 128))  # Màu vàng nửa trong suốt
    highlight_images["legal"].fill((0, 255, 0, 128))       # Màu xanh lá nửa trong suốt
    highlight_images["illegal"].fill((255, 0, 0, 128))     # Màu đỏ nửa trong suốt
    highlight_images["check"].fill((255, 0, 255, 128))     # Màu tím nửa trong suốt
    
    return board_image, piece_images, highlight_images

# Hàm tiện ích
def should_flip_board(player_side):
    """Kiểm tra xem có nên đảo chiều bàn cờ không."""
    return player_side == "black"

# Vẽ phần đế dùng chung
def draw_board_base(screen, board_image, panel_width):
    """Vẽ ảnh bàn cờ cơ bản"""
    # Vẽ bàn cờ
    screen.blit(board_image, (panel_width, 0))
    
    # Vẽ ký tự chuẩn cho hàng (1-8) và cột (a-h)
    font = pygame.font.SysFont("Arial", 36, bold=True)
    
    for i in range(8):
        # Số hàng (1-8)
        label = font.render(str(8 - i), True, (40, 40, 40))
        x = panel_width + 8
        y = i * SQUARE_SIZE + SQUARE_SIZE - label.get_height() - 8
        screen.blit(label, (x, y))
        
        # Chữ cột (a-h)
        label = font.render(chr(ord('a') + i), True, (40, 40, 40))
        x = panel_width + i * SQUARE_SIZE + SQUARE_SIZE - label.get_width() - 8
        y = BOARD_SIZE - label.get_height() - 8
        screen.blit(label, (x, y))

# Vẽ quân cờ
def draw_pieces(screen, board, piece_images, panel_width, flip_board=False):
    """Vẽ quân cờ lên bàn cờ"""
    for row in range(8):
        for col in range(8):
            if flip_board:
                square = chess.square(7-col, row)
            else:
                square = chess.square(col, 7-row)
                
            piece = board.piece_at(square)
            if piece:
                piece_image = piece_images.get(piece.symbol())
                if piece_image:
                    screen.blit(piece_image, 
                              (panel_width + col * SQUARE_SIZE, row * SQUARE_SIZE))

# Vẽ highlight cho các ô
def draw_highlights(screen, board, selected_square, legal_moves, highlight_images, panel_width, flip_board=False):
    """Vẽ highlight cho các ô được chọn và nước đi hợp lệ"""
    # Highlight king in check
    if board.is_check():
        king_color = board.turn
        king_square = board.king(king_color)
        if king_square is not None:
            if flip_board:
                col = 7 - chess.square_file(king_square)
                row = chess.square_rank(king_square)
            else:
                col = chess.square_file(king_square)
                row = 7 - chess.square_rank(king_square)
            screen.blit(highlight_images["check"], 
                       (panel_width + col * SQUARE_SIZE, row * SQUARE_SIZE))

    # Highlight selected piece
    if selected_square is not None:
        if flip_board:
            col = 7 - chess.square_file(selected_square)
            row = chess.square_rank(selected_square)
        else:
            col = chess.square_file(selected_square)
            row = 7 - chess.square_rank(selected_square)
        screen.blit(highlight_images["selected"], 
                   (panel_width + col * SQUARE_SIZE, row * SQUARE_SIZE))

        # Highlight legal moves
        for move, is_capture in legal_moves:
            if flip_board:
                col = 7 - chess.square_file(move.to_square)
                row = chess.square_rank(move.to_square)
            else:
                col = chess.square_file(move.to_square)
                row = 7 - chess.square_rank(move.to_square)
            highlight = highlight_images["illegal"] if is_capture else highlight_images["legal"]
            screen.blit(highlight, 
                       (panel_width + col * SQUARE_SIZE, row * SQUARE_SIZE))

# Vẽ thông báo kết thúc game
def draw_end_game_message(screen, message):
    """Vẽ thông báo kết thúc game"""
    # Tạo fonts
    title_font = pygame.font.Font(None, 64)
    instruction_font = pygame.font.Font(None, 48)
    
    # Tạo text surfaces
    title_surface = title_font.render(message, True, WHITE)
    instruction_surface = instruction_font.render("Press SPACE to return to menu", True, WHITE)
    
    # Position text
    title_rect = title_surface.get_rect()
    title_rect.center = (WIDTH // 2, HEIGHT // 2 - 40)
    
    instruction_rect = instruction_surface.get_rect()
    instruction_rect.center = (WIDTH // 2, HEIGHT // 2 + 40)
    
    # Tạo overlay trong suốt
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(160)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))
    
    # Vẽ hộp thông báo
    padding = 60
    background_rect = pygame.Rect(0, 0, 
                                max(title_rect.width, instruction_rect.width) + padding * 2,
                                title_rect.height + instruction_rect.height + padding * 2)
    background_rect.center = (WIDTH // 2, HEIGHT // 2)
    
    pygame.draw.rect(screen, (50, 50, 50), background_rect)
    pygame.draw.rect(screen, WHITE, background_rect, 4)
    
    # Vẽ thông báo
    screen.blit(title_surface, title_rect)
    screen.blit(instruction_surface, instruction_rect)

# Vẽ nút chung
def draw_button(screen, rect, text, bg_color=GRAY, text_color=BLACK, font_size=32):
    """Vẽ nút với văn bản ở giữa"""
    pygame.draw.rect(screen, bg_color, rect)
    pygame.draw.rect(screen, DARK_GRAY, rect, 3)
    
    font = pygame.font.SysFont("Arial", font_size)
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)
    
    return rect  # Trả về rect để kiểm tra va chạm

# Vẽ menu chung
def draw_menu(screen, title, options, selected=None):
    """Vẽ menu với các tùy chọn"""
    screen.fill(WHITE)
    
    # Vẽ tiêu đề
    title_font = pygame.font.SysFont("Arial", 48, bold=True)
    title_text = title_font.render(title, True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 80))
    
    # Vẽ các tùy chọn
    buttons = []
    option_font = pygame.font.SysFont("Arial", 36)
    
    for i, (option_text, option_value) in enumerate(options):
        y_pos = 240 + i * 120
        button_rect = pygame.Rect(WIDTH // 2 - 200, y_pos, 400, 80)
        
        # Highlight option đang chọn
        color = LIGHT_BLUE if option_value == selected else GRAY
        
        pygame.draw.rect(screen, color, button_rect)
        pygame.draw.rect(screen, DARK_GRAY, button_rect, 3)
        
        text = option_font.render(option_text, True, BLACK)
        screen.blit(text, (button_rect.x + button_rect.width // 2 - text.get_width() // 2,
                          button_rect.y + button_rect.height // 2 - text.get_height() // 2))
        
        buttons.append((button_rect, option_value))
    
    # Vẽ nút Back nếu cần
    if selected is not None:
        back_rect = pygame.Rect(20, HEIGHT - 80, 120, 60)
        pygame.draw.rect(screen, GRAY, back_rect)
        pygame.draw.rect(screen, DARK_GRAY, back_rect, 3)
        
        back_text = pygame.font.SysFont("Arial", 28).render("Back", True, BLACK)
        screen.blit(back_text, (back_rect.x + back_rect.width // 2 - back_text.get_width() // 2,
                               back_rect.y + back_rect.height // 2 - back_text.get_height() // 2))
        
        buttons.append((back_rect, "back"))
    
    return buttons