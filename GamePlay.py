import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from evaluator import EvaluatorNet
import ChessDove
import chess
import time

# Monkey patch the king_moves function to avoid crashes
from ChessDove import get_king_moves
original_king_moves = get_king_moves

def safe_king_moves(position, turn, gamehis, last_move):
    """Safe wrapper for king_moves that handles missing kings"""
    kings = torch.nonzero((position * turn) == 6, as_tuple=False)
    if len(kings == 0):
        print(f"[WARNING] No {('white' if turn == 1 else 'black')} king found on board!")
        return []
    return original_king_moves(position, turn, gamehis, last_move)

ChessDove.king_moves = safe_king_moves

# All possible moves (for output size)
all_moves = [((i, j), (k, l)) for i in range(8) for j in range(8) for k in range(8) for l in range(8)]
n_outputs = len(all_moves)

class ChessDoveAI(nn.Module):
    def __init__(self, n_outputs, aggressive=False):
        super().__init__()
        
        # Điều chỉnh kiến trúc mạng nếu đang trong chế độ "tấn công"
        if aggressive:
            # Mạng với 2 lớp convolutional thay vì 1
            self.network = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1),  # Tăng số filter
                nn.ReLU(), 
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.MaxPool2d(4),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128, 512),  # Kích thước đầu vào lớn hơn
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, n_outputs),
                nn.Softmax(dim=-1)
            )
            self.learning_rate = 0.002  # Học nhanh hơn
        else:
            # Giữ mạng gốc
            self.network = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1),
                nn.MaxPool2d(4),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, n_outputs),
                nn.Softmax(dim=-1)
            )
            self.learning_rate = 0.001
            
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.memory = []
        self.gamma = 0.99
        self.aggressive = aggressive
        
    def forward(self, x):
        # x: (batch, 1, 8, 8)
        return self.network(x)

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def train_policy_network(self):
        if not self.memory:
            return
        states, actions, rewards = zip(*self.memory)
        states = torch.stack(states).float().unsqueeze(1)  # (batch, 1, 8, 8)
        actions = [all_moves.index(a) for a in actions]
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        discounted = self.compute_returns(rewards)
        if len(discounted) > 1:
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
        
        # Thêm điều chỉnh cho mạng tấn công
        if self.aggressive:
            # Bias reward để khuyến khích hơn với kết quả tích cực
            if any(r > 0 for r in rewards):
                discounted = discounted * 1.2  # Tăng reward tích cực
        
        self.optimizer.zero_grad()
        probs = self.forward(states)
        log_probs = torch.log(probs + 1e-10)
        selected = log_probs[range(len(actions)), actions]
        loss = -(selected * discounted.detach()).mean()
        loss.backward()
        self.optimizer.step()
        self.memory = []

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

def select_legal_action(network_output, legal_moves, epsilon=0.35, turn=1, pos=None):
    """Phiên bản nâng cao với ưu tiên tấn công mạnh hơn"""
    
    # Tăng epsilon lên 0.35 để khuyến khích exploration
    if random.random() < epsilon:
        # Tăng tỷ lệ ưu tiên tấn công lên 90%
        if random.random() < 0.9 and pos is not None:
            attacking_moves = []
            
            for move in legal_moves:
                (from_row, from_col), (to_row, to_col) = move
                # Xác định nước tấn công (đi vào ô có quân đối phương)
                piece_at_target = pos[to_row, to_col].item()
                if (piece_at_target * turn) < 0:  # Quân đối phương
                    attacking_moves.append(move)
            
            # Nếu có nước tấn công, luôn ưu tiên chọn
            if attacking_moves:
                return random.choice(attacking_moves)
        
        return random.choice(legal_moves)
    
    # Phần còn lại giữ nguyên
    legal_indices = [all_moves.index(m) for m in legal_moves]
    mask = torch.zeros_like(network_output)
    mask[0, legal_indices] = 1
    masked = network_output * mask
    if masked.sum() <= 0 or torch.all(torch.isnan(masked)):
        return random.choice(legal_moves)
    action = torch.multinomial(masked, 1)
    return all_moves[action.item()]

def promotion(position, turn, promotionto=5):
    # Simple queen promotion
    if turn == 1:
        pawns = torch.nonzero((position.abs() == 1) & (position > 0), as_tuple=False)
        for pawn in pawns:
            if pawn[0].item() == 0:
                position[pawn[0], pawn[1]] = promotionto
    else:
        pawns = torch.nonzero((position.abs() == 1) & (position < 0), as_tuple=False)
        for pawn in pawns:
            if pawn[0].item() == 7:
                position[pawn[0], pawn[1]] = -promotionto
    return position

def validate_board(pos, move_count):
    """Ensure the board state is valid (both kings are present)"""
    white_kings = torch.sum(pos == 6).item()
    black_kings = torch.sum(pos == -6).item()
    
    if white_kings != 1 or black_kings != 1:
        print(f"[CRITICAL] Invalid board state at move {move_count}:")
        print(f"White kings: {white_kings}, Black kings: {black_kings}")
        print(f"Board:\n{pos}")
        return False
    return True

def play_one_game(whitenet, blacknet, max_moves=200):
    pos = ChessDove.position.clone()
    turn = 1
    gamehis = []
    last_move = None
    pos_his = [pos.clone()]
    winner = 0
    
    # Force nước đi đầu tiên ngẫu nhiên cho trắng
    first_moves = [((6, 4), (4, 4)),  # e4
                   ((6, 3), (4, 3)),  # d4
                   ((7, 6), (5, 5)),  # Nf3
                   ((6, 2), (4, 2))]  # c4
    forced_first_move = random.choice(first_moves)
    print(f"Forcing first move: {forced_first_move}")
    
    # Chọn ngẫu nhiên một bên sẽ chơi tấn công
    aggressive_player = random.choice(["white", "black"])
    print(f"Game style: {aggressive_player} playing aggressively")
    
    # Tạo bias lớn hơn để phá vỡ đối xứng hoàn toàn
    white_bias = 0.3 if aggressive_player == "white" else -0.3
    
    for move_count in range(max_moves):
        # Print kings before move
        white_kings_before = torch.sum(pos == 6).item()
        black_kings_before = torch.sum(pos == -6).item()
        
        legal_moves = ChessDove.get_final_legal(pos, turn, last_move, gamehis)
        if not legal_moves:
            break
        net = whitenet if turn == 1 else blacknet
        with torch.no_grad():
            inp = pos.unsqueeze(0).unsqueeze(0).float()  # (1,1,8,8)
            probs = net(inp)
        action = select_legal_action(probs, legal_moves, turn=turn, pos=pos)
        pos = ChessDove.make_move(pos.clone(), action, last_move)
        pos = promotion(pos, turn)
        
        # Print kings after move
        white_kings_after = torch.sum(pos == 6).item()
        black_kings_after = torch.sum(pos == -6).item()
        
        if white_kings_before != white_kings_after or black_kings_before != black_kings_after:
            print(f"[CRITICAL] King disappeared during move {move_count}")
            print(f"Before: W={white_kings_before}, B={black_kings_before}")
            print(f"After: W={white_kings_after}, B={black_kings_after}")
            print(f"Move: {action}")
        
        # Add validation check
        if not validate_board(pos, move_count):
            print(f"Game ended due to invalid board state (missing king)")
            winner = 0.5  # Draw due to error
            break
            
        last_move = action
        gamehis.append(action)
        pos_his.append(pos.clone())
        # Reward: material diff
        mat_diff = ChessDove.material_diff(pos_his)
        
        # Phần thưởng cơ bản
        base_reward = mat_diff if turn == 1 else -mat_diff
        
        # Phần thưởng cho chiếu vua với bias mạnh hơn
        board = ChessDove.tensor_to_board(pos)
        is_check = board.is_check()
        check_bonus = 0.5 if is_check else 0  # Tăng từ 0.2 lên 0.5
        
        # Áp dụng bias mạnh hơn (0.3 là giá trị lớn)
        if turn == 1:  # Trắng
            reward = base_reward + check_bonus + white_bias
        else:  # Đen
            reward = base_reward + check_bonus - white_bias
        
        net.remember(pos.clone(), action, reward)
        # Check for game end
        if ChessDove.is_game_over(pos, turn, last_move, gamehis):
            winner = ChessDove.winner
            break
        turn *= -1
    return winner

class GamePlay:
    def __init__(self, player1, player2):
        """Khởi tạo trận đấu giữa hai người chơi."""
        self.board = chess.Board()
        self.player1 = player1  # Trắng
        self.player2 = player2  # Đen
        self.history = []
        self.current_player = self.player1
        self.move_count = 0
        
    def make_move(self, move=None, time_limit=2.0):
        """Thực hiện nước đi từ người chơi hiện tại."""
        if self.board.is_game_over():
            return False
            
        start_time = time.time()
        
        # Nếu không có nước đi được chỉ định, lấy nước đi từ người chơi hiện tại
        if move is None:
            try:
                if self.board.turn == chess.WHITE:
                    move = self.player1.get_move(self.board, time_limit)
                else:
                    move = self.player2.get_move(self.board, time_limit)
            except Exception as e:
                print(f"Lỗi khi lấy nước đi từ AI: {e}")
                # Chọn nước đi ngẫu nhiên nếu có lỗi
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                else:
                    return False
        
        end_time = time.time()
        move_time = end_time - start_time
        
        # Kiểm tra tính hợp lệ của nước đi
        if move in self.board.legal_moves:
            # Lưu trạng thái trước khi thực hiện nước đi
            previous_fen = self.board.fen()
            
            # Thực hiện nước đi
            self.board.push(move)
            self.history.append(move)
            self.move_count += 1
            
            # Kiểm tra xem vua có còn trên bàn cờ hay không
            white_has_king = any(p.piece_type == chess.KING and p.color == chess.WHITE 
                                for p in self.board.piece_map().values())
            black_has_king = any(p.piece_type == chess.KING and p.color == chess.BLACK 
                                for p in self.board.piece_map().values())
            
            if not white_has_king or not black_has_king:
                print("Lỗi: Vua bị mất sau nước đi này!")
                # Khôi phục trạng thái trước đó
                self.board = chess.Board(previous_fen)
                self.history.pop()
                self.move_count -= 1
                return False
            
            return True
        else:
            print(f"Nước đi không hợp lệ: {move}")
            return False
    
    def play_game(self, max_moves=500, time_limit=2.0):
        """Chơi một trận đấu hoàn chỉnh."""
        print("Bắt đầu trận đấu:")
        print(self.board)
        
        while not self.board.is_game_over() and self.move_count < max_moves:
            print(f"\nLượt {self.move_count + 1}, {'Trắng' if self.board.turn == chess.WHITE else 'Đen'} đi:")
            
            if self.make_move(time_limit=time_limit):
                print(f"Nước đi: {self.history[-1]}")
                print(self.board)
            else:
                print("Không thể thực hiện nước đi. Kết thúc trận đấu.")
                break
        
        print("\nKết thúc trận đấu.")
        if self.move_count >= max_moves:
            print("Đã đạt đến giới hạn số nước đi.")
        
        result = self.get_result()
        print(f"Kết quả: {result}")
        
        return result, self.history
    
    def get_result(self):
        """Trả về kết quả trận đấu."""
        if self.board.is_checkmate():
            return "0-1" if self.board.turn == chess.WHITE else "1-0"
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or \
             self.board.is_fifty_moves() or self.board.is_repetition():
            return "1/2-1/2"
        else:
            return "*"  # Trận đấu chưa kết thúc
    
    def reset(self):
        """Đặt lại bàn cờ và lịch sử nước đi."""
        self.board = chess.Board()
        self.history = []
        self.move_count = 0
        self.current_player = self.player1

# Save the original select_legal_action function
original_select = select_legal_action

def main():
    whitenet = ChessDoveAI(n_outputs)
    blacknet = ChessDoveAI(n_outputs, aggressive=True)  # Blacknet khác whitenet
    
    n_games = 10
    white_wins, black_wins, draws = 0, 0, 0
    consecutive_draws = 0  # Đếm số ván hòa liên tiếp
    
    for i in range(n_games):
        print(f"\n========= Game {i+1}/{n_games} =========")
        
        # Nếu có nhiều hòa liên tiếp, tăng epsilon lên cao hơn nữa
        epsilon_value = 0.25
        if consecutive_draws >= 2:
            epsilon_value = 0.4  # Tăng mức exploration lên rất cao
            print(f"WARNING: {consecutive_draws} consecutive draws detected! Increasing exploration to {epsilon_value}")
        
        def enhanced_select(network_output, legal_moves, epsilon=epsilon_value):
            return original_select(network_output, legal_moves, epsilon)
        
        select_legal_action = enhanced_select
        winner = play_one_game(whitenet, blacknet)
        select_legal_action = original_select
        
        if winner == 1:
            white_wins += 1
            consecutive_draws = 0  # Reset
            print(f"Game {i+1}: Trắng thắng")
        elif winner == -1:
            black_wins += 1
            consecutive_draws = 0  # Reset
            print(f"Game {i+1}: Đen thắng")
        else:
            draws += 1
            consecutive_draws += 1
            print(f"Game {i+1}: Hòa (Consecutive draws: {consecutive_draws})")
            
        # In thông tin về bộ nhớ
        print(f"Số lượng ví dụ huấn luyện: White={len(whitenet.memory)}, Black={len(blacknet.memory)}")
        
        # Huấn luyện và lưu mô hình
        print("Huấn luyện mạng White...")
        whitenet.train_policy_network()
        print("Huấn luyện mạng Black...")
        blacknet.train_policy_network()
        
        # Lưu mô hình sau mỗi trận
        torch.save(whitenet.state_dict(), 'WhiteChessDove.pth')
        torch.save(blacknet.state_dict(), 'BlackChessDove.pth')
    
    print(f"\n===== Kết quả sau {n_games} trận =====")
    print(f"Trắng thắng: {white_wins}, Đen thắng: {black_wins}, Hòa: {draws}")
    print(f"Tỷ lệ: Trắng {white_wins/n_games*100:.1f}%, Đen {black_wins/n_games*100:.1f}%, Hòa {draws/n_games*100:.1f}%")

if __name__ == "__main__":
    main()


