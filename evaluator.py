import torch
import torch.nn as nn
import torch.nn.functional as F

class EvaluatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)
        
        # Lưu thêm thông tin về độ sâu tìm kiếm mặc định
        self.depth = 3
    
    def forward(self, board_tensor):
        # Xử lý đầu vào để đảm bảo định dạng đúng: [batch_size, 1, 8, 8]
        if board_tensor.dim() == 2:  # Input chỉ là bàn cờ 8x8
            board_tensor = board_tensor.unsqueeze(0).unsqueeze(0)
        elif board_tensor.dim() == 3:  # Input là batch_size x 8 x 8
            board_tensor = board_tensor.unsqueeze(1)
        
        # Chuyển đổi sang kiểu float32
        x = board_tensor.float()
        
        # Áp dụng các lớp convolutional
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Làm phẳng tensor
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Áp dụng các lớp fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Áp dụng tanh để giới hạn đầu ra trong khoảng [-1, 1]
        return torch.tanh(x)
    
    def get_depth(self):
        """Trả về độ sâu tìm kiếm được khuyến nghị"""
        return self.depth

def create_initial_model():
    """Tạo mô hình ban đầu và lưu nó"""
    model = EvaluatorNet()
    torch.save(model.state_dict(), 'best_eval.pt')
    print("Created and saved initial model to best_eval.pt")
    return model

if __name__ == "__main__":
    create_initial_model()