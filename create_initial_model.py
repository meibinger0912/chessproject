import torch
import os
from evaluator import EvaluatorNet

def create_empty_model():
    """Tạo một mô hình rỗng để bắt đầu quá trình huấn luyện"""
    print("Creating initial evaluator model...")
    model = EvaluatorNet()
    
    # Lưu mô hình
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_eval.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Initial model created and saved to {save_path}")

if __name__ == "__main__":
    create_empty_model()