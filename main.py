import pygame
import sys
import subprocess
from graphic import WIDTH, HEIGHT, WHITE, BLACK, GRAY, draw_menu

# Đường dẫn Stockfish
STOCKFISH_PATH = "D:/Trí tuệ nhân tạo/chess project/stockfish/stockfish.exe"

# Khởi tạo pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Chess Game')

def main():
    """Hàm main điều phối các màn hình và luồng game"""
    state = "mode_selection"  # Trạng thái ban đầu: chọn chế độ chơi
    
    # Lưu trữ lựa chọn của người chơi
    player_choices = {
        "ai_type": None,       # "stockfish" hoặc "chessdove"
        "elo": None,           # Elo cho Stockfish
        "player_side": None    # "white" hoặc "black"
    }
    
    # Vòng lặp chính của chương trình
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Xử lý sự kiện chung
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Xử lý theo trạng thái hiện tại
                if state == "mode_selection":
                    # Vẽ và kiểm tra nút chọn chế độ
                    buttons = draw_mode_selection()
                    for button, mode in buttons:
                        if button.collidepoint(mouse_pos):
                            if mode == "human_vs_ai":
                                state = "ai_selection"
                            elif mode == "ai_training":
                                # Chạy file watch_ai_battle.py
                                subprocess.Popen(["python", "D:/Trí tuệ nhân tạo/chess project/ChessDove/watch_ai_battle.py"])
                                pygame.quit()
                                sys.exit()
                
                elif state == "ai_selection":
                    # Vẽ và kiểm tra nút chọn AI
                    buttons = draw_ai_selection()
                    for button, action in buttons:
                        if button.collidepoint(mouse_pos):
                            if action == "stockfish":
                                player_choices["ai_type"] = "stockfish"
                                state = "elo_selection"
                            elif action == "chessdove":
                                player_choices["ai_type"] = "chessdove"
                                state = "side_selection"
                            elif action == "back":
                                state = "mode_selection"
                
                elif state == "elo_selection":
                    # Import từ gui.py
                    from gui import draw_elo_selection
                    buttons = draw_elo_selection(screen, WIDTH, HEIGHT)
                    
                    for button, elo in buttons:
                        if button.collidepoint(mouse_pos):
                            if elo == "back":
                                state = "ai_selection"
                            else:
                                player_choices["elo"] = elo
                                state = "side_selection"
                
                elif state == "side_selection":
                    # Import dựa trên AI type
                    if player_choices["ai_type"] == "stockfish":
                        from gui import draw_side_selection
                        buttons = draw_side_selection(screen, WIDTH, HEIGHT)
                    else:
                        from guix import draw_chessdove_info as draw_side_selection
                        buttons = draw_side_selection(screen)  # Hàm draw_chessdove_info chỉ cần tham số screen
                    
                    for button, side in buttons:
                        if button.collidepoint(mouse_pos):
                            if side == "white" or side == "black":
                                player_choices["player_side"] = side
                                
                                # Bắt đầu trò chơi với các lựa chọn đã có
                                start_game(
                                    player_choices["ai_type"],
                                    player_choices["elo"],
                                    player_choices["player_side"]
                                )
                                
                                # Reset và trở về menu chính sau khi chơi xong
                                state = "mode_selection"
                                player_choices = {
                                    "ai_type": None,
                                    "elo": None,
                                    "player_side": None
                                }
                                
                            elif side == "back":
                                if player_choices["ai_type"] == "stockfish":
                                    state = "elo_selection"
                                else:
                                    state = "ai_selection"
        
        # Vẽ màn hình theo trạng thái hiện tại
        if state == "mode_selection":
            draw_mode_selection()
        elif state == "ai_selection":
            draw_ai_selection()
        elif state == "elo_selection":
            from gui import draw_elo_selection
            draw_elo_selection(screen, WIDTH, HEIGHT)
        elif state == "side_selection":
            if player_choices["ai_type"] == "stockfish":
                from gui import draw_side_selection
                draw_side_selection(screen, WIDTH, HEIGHT)
            else:
                from guix import draw_chessdove_info
                draw_chessdove_info(screen)
        
        pygame.display.flip()
        clock.tick(30)  # 30 FPS
    
    pygame.quit()

def draw_mode_selection():
    """Vẽ màn hình chọn chế độ chơi."""
    options = [
        ("AI vs Human", "human_vs_ai"),
        ("AI Training", "ai_training")
    ]
    return draw_menu(screen, "Choose Game Mode", options)

def draw_ai_selection():
    """Vẽ màn hình chọn AI."""
    options = [
        ("Stockfish", "stockfish"),
        ("ChessDove AI", "chessdove")
    ]
    return draw_menu(screen, "Choose AI Engine", options, None)

def start_game(ai_type, elo_rating, player_side):
    """Bắt đầu trò chơi với các tham số đã chọn"""
    if ai_type == "stockfish":
        from gui import run_stockfish_game
        run_stockfish_game(elo_rating, player_side, STOCKFISH_PATH)
    else:
        from guix import run_chessdove_game
        run_chessdove_game(player_side, STOCKFISH_PATH)

if __name__ == "__main__":
    main()
