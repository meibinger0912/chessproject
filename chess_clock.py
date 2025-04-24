import time
import chess

class ChessClock:
    def __init__(self):
        self.game_time = 0  # Total game time in seconds
        self.move_time = 0  # Current move time in seconds
        self.white_time = 0  # White's total thinking time
        self.black_time = 0  # Black's total thinking time
        self.current_player = chess.WHITE
        self.last_update = None
        self.move_start = None
        self.is_running = False
        self.MOVE_LIMIT = 10  # 10 seconds per move
        self.game_over = False
        self.result_message = None
        self.last_move_by_ai = False

    def start(self):
        """Start the game clock"""
        self.is_running = True
        current = time.time()
        self.last_update = current
        self.move_start = current

    def stop(self):
        """Stop the clock"""
        self.is_running = False
        self.last_update = None
        self.move_start = None

    def update(self):
        """Update all time values and check game ending"""
        if not self.is_running or self.last_update is None:
            return True

        current = time.time()
        elapsed = current - self.last_update
        move_elapsed = current - self.move_start

        # Update times
        self.game_time += elapsed
        self.move_time = move_elapsed

        if self.current_player == chess.WHITE:
            self.white_time += elapsed
        else:
            self.black_time += elapsed

        self.last_update = current

        # Debug info
        if self.move_time > 8:
            print(f"Debug time check:")
            print(f"Current turn: {'Black' if self.current_player == chess.BLACK else 'White'}")
            print(f"Move time: {self.move_time:.1f}s")
            print(f"Last move by AI: {self.last_move_by_ai}")

        # Chỉ kiểm tra thời gian cho người chơi, không kiểm tra cho AI
        if self.move_time > self.MOVE_LIMIT and not self.last_move_by_ai:
            self.game_over = True
            current_player = "Black" if self.current_player == chess.BLACK else "White"
            self.result_message = f"Time's up! {current_player} loses!"
            return False

        return True

    def switch_player(self):
        """Switch active player and reset move timer"""
        self.current_player = not self.current_player
        self.move_start = time.time()
        self.move_time = 0
        self.last_move_by_ai = False  # Reset AI move flag

    def format_time(self, seconds):
        """Format time as MM:SS"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"