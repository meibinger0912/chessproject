import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import numpy as np
from datetime import datetime
from evaluator import EvaluatorNet
from torch.utils.data import DataLoader, TensorDataset, random_split

# Cấu hình huấn luyện
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
EPOCHS = 50  # Giảm số epoch nếu muốn huấn luyện nhanh hơn
VALIDATION_SPLIT = 0.2  # 20% dữ liệu dùng làm validation
DATA_DIR = "training_data"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_training_data():
    """Load toàn bộ dữ liệu huấn luyện từ thư mục training_data"""
    print("Loading training data...")
    
    # Tìm tất cả các file states*.pt
    state_files = glob.glob(os.path.join(DATA_DIR, "states_*.pt"))
    if not state_files:
        print(f"No training data found in {DATA_DIR}")
        return None, None
        
    # Load và nối tất cả dữ liệu
    all_states = []
    all_values = []
    
    for state_file in state_files:
        # Tìm file values tương ứng
        basename = os.path.basename(state_file)
        game_id = basename.split("_")[-1]  # Lấy phần game{id}.pt
        value_file = os.path.join(DATA_DIR, f"values_{basename.split('_')[1]}_{basename.split('_')[2]}_{game_id}")
        
        if not os.path.exists(value_file):
            print(f"Warning: Values file not found for {basename}")
            continue
            
        try:
            states = torch.load(state_file)
            values = torch.load(value_file)
            
            if len(states) != len(values):
                print(f"Warning: Mismatch in states and values length for {basename}")
                continue
                
            all_states.append(states)
            all_values.append(values)
        except Exception as e:
            print(f"Error loading {basename}: {e}")
            
    if not all_states:
        print("No valid data found")
        return None, None
        
    # Nối dữ liệu từ nhiều file
    all_states = torch.cat(all_states)
    all_values = torch.cat(all_values)
    
    print(f"Loaded {len(all_states)} training examples")
    return all_states, all_values

def train_model():
    """Huấn luyện mạng neural EvaluatorNet"""
    # 1. Load dữ liệu
    states, values = load_training_data()
    if states is None:
        # Nếu không có dữ liệu thật, tạo dữ liệu giả
        print("No real training data found. Creating synthetic data for initial training...")
        states, values = create_synthetic_data(5000)
    
    # 2. Chia validation set
    dataset = TensorDataset(states, values)
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 3. Khởi tạo model và optimizer
    model = EvaluatorNet()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 4. Huấn luyện mô hình
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.float()).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # In tiến trình
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data.float()).squeeze()
                val_loss += criterion(output, target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} complete - Train loss: {avg_train_loss:.6f}, Validation loss: {avg_val_loss:.6f}")
        
        # Lưu model tốt nhất
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_eval.pt')
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
    
    print("Training complete!")
    return model

def create_synthetic_data(num_samples=1000):
    """Tạo dữ liệu giả để huấn luyện ban đầu"""
    import random
    
    def material_evaluation(board_tensor):
        """Hàm đánh giá tĩnh dựa trên vật chất"""
        piece_values = {
            1: 1.0,   # Pawn
            2: 5.0,   # Rook
            3: 3.0,   # Knight
            4: 3.0,   # Bishop
            5: 9.0,   # Queen
            6: 0.0    # King - không tính giá trị
        }
        
        # Tính tổng giá trị
        value = 0.0
        for i in range(8):
            for j in range(8):
                piece = board_tensor[i][j].item()
                if piece != 0:  # Nếu có quân cờ
                    sign = 1 if piece > 0 else -1
                    value += sign * piece_values[abs(piece)]
        
        # Chuẩn hóa về [-1, 1]
        return torch.tanh(torch.tensor(value / 30.0))

    # Tạo các bàn cờ ngẫu nhiên
    boards = []
    values = []
    
    for _ in range(num_samples):
        board = torch.zeros((8, 8), dtype=torch.float32)
        
        # Luôn có vua
        board[random.randint(0, 7)][random.randint(0, 7)] = 6  # Vua trắng
        board[random.randint(0, 7)][random.randint(0, 7)] = -6  # Vua đen
        
        # Thêm quân ngẫu nhiên
        num_pieces = random.randint(5, 20)
        for _ in range(num_pieces):
            i, j = random.randint(0, 7), random.randint(0, 7)
            if board[i][j] == 0:  # chỉ đặt nếu ô trống
                piece = random.choice([1, 2, 3, 4, 5, -1, -2, -3, -4, -5])
                board[i][j] = piece
        
        # Tính giá trị bằng hàm đánh giá tĩnh
        value = material_evaluation(board)
        
        boards.append(board)
        values.append(value)
    
    return torch.stack(boards), torch.tensor(values)

if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Train model
    model = train_model()
    
    print("Final model saved as best_eval.pt")