import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
import json
import torch
import re

def normalize_path(path):
    """Chuẩn hóa đường dẫn để phù hợp với mọi hệ thống"""
    return os.path.normpath(os.path.abspath(path))

# Define paths - Cập nhật các đường dẫn thư mục
DOVE_DIR = normalize_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ChessDove"))
STOCKFISH_PATH = "D:\Trí tuệ nhân tạo\chess project\stockfish\stockfish.exe"
# Add paths to Python path
sys.path.insert(0, DOVE_DIR)  # Insert at beginning to ensure it's checked first
sys.path.insert(0, os.path.dirname(DOVE_DIR))

# Try importing EvaluatorNet with better error handling
try:
    from ChessDove.evaluator import EvaluatorNet  # Ensure the correct path to evaluator
    print(f"Successfully imported EvaluatorNet from {DOVE_DIR}")
except ImportError as e:
    print(f"Could not import evaluator directly: {e}")
    try:
        # Thử lại với đường dẫn khác nếu cần
        from ChessDove.evaluator import EvaluatorNet
        print("Successfully imported EvaluatorNet from ChessDove package")
    except ImportError as e:
        print(f"Could not import from ChessDove package: {e}")
        raise ImportError(f"Could not import EvaluatorNet. Please ensure evaluator.py exists in {DOVE_DIR}")

def log_message(message):
    """Ghi log với timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    with open("training_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def run_command(command, description):
    """Chạy lệnh và log kết quả"""
    log_message(f"Bắt đầu: {description}")
    log_message(f"Lệnh thực thi: {command}")
    start_time = time.time()
    try:
        # Thêm encoding và chỉ định môi trường để hỗ trợ tiếng Việt
        my_env = os.environ.copy()
        my_env['PYTHONIOENCODING'] = 'utf-8'
        
        # Thêm os.environ cho các biến môi trường
        os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True,
            env=my_env,
            encoding='utf-8'  # Thêm encoding parameter
        )
        
        # Print output theo dòng
        output_text = ""
        for line in process.stdout:
            cleaned_line = line.strip()
            print(cleaned_line)
            output_text += cleaned_line + "\n"
        
        process.wait()
        elapsed_time = time.time() - start_time
        log_message(f"Hoàn thành: {description} - Thời gian: {elapsed_time:.2f}s")
        
        # Lưu output để debug
        with open(f"debug_{description.replace(' ', '_')}.log", "w", encoding="utf-8") as f:
            f.write(output_text)
        
        if process.returncode != 0:
            log_message(f"LỖI: {description} trả về mã lỗi {process.returncode}")
            return False
        return True
    except Exception as e:
        log_message(f"NGOẠI LỆ: {description} gặp lỗi: {str(e)}")
        return False

def check_stockfish():
    """Kiểm tra Stockfish có tồn tại không"""
    if not os.path.exists(STOCKFISH_PATH):
        log_message(f"LỖI: Không tìm thấy Stockfish tại {STOCKFISH_PATH}")
        log_message("Vui lòng tải Stockfish và đặt vào thư mục stockfish")
        return False
    return True

def update_stockfish_depth(depth):
    """Cập nhật độ sâu Stockfish trong file play_vs_stockfish.py"""
    filepath = os.path.join(DOVE_DIR, "play_vs_stockfish.py")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Tìm và thay thế giá trị độ sâu
        pattern = r"stockfish_depth = \d+"
        replacement = f"stockfish_depth = {depth}"
        new_content = re.sub(pattern, replacement, content)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        log_message(f"Đã cập nhật stockfish_depth thành {depth}")
        return True
    except Exception as e:
        log_message(f"Lỗi khi cập nhật độ sâu Stockfish: {str(e)}")
        return False

def train_from_stockfish():
    """Huấn luyện từ Stockfish (Supervised Learning)"""
    if not check_stockfish():
        return False
    
    # 1. Thu thập dữ liệu từ Stockfish
    stockfish_cmd = f"cd {DOVE_DIR} && python play_vs_stockfish.py"
    if not run_command(stockfish_cmd, "Thu thập dữ liệu từ Stockfish"):
        return False
    
    # 2. Huấn luyện mạng neural đánh giá
    train_cmd = f"cd {DOVE_DIR} && python train_evaluator.py"
    if not run_command(train_cmd, "Huấn luyện mạng đánh giá"):
        return False
    
    log_message("Hoàn thành quá trình học từ Stockfish!")
    return True

def run_self_play():
    """Chạy self-play (Reinforcement Learning)"""
    # Chạy GamePlay.py để cải thiện mô hình qua self-play
    gameplay_cmd = f"cd {DOVE_DIR} && python GamePlay.py"
    return run_command(gameplay_cmd, "Tự học qua Self-play")

def evaluate_elo(stockfish_depth=10):
    """Đánh giá Elo thông qua script riêng biệt"""
    log_message(f"Đánh giá Elo với Stockfish độ sâu {stockfish_depth}")
    
    if not check_stockfish():
        log_message("Không tìm thấy Stockfish, bỏ qua đánh giá Elo")
        return False, 0
    
    # Chạy script đánh giá riêng biệt
    elo_cmd = f"cd {DOVE_DIR} && python evaluate_elo.py --stockfish-path=\"{STOCKFISH_PATH}\" --stockfish-depth={stockfish_depth} --num-games=30"
    success = run_command(elo_cmd, "Đánh giá Elo")
    
    # Ngay cả khi không thành công, vẫn thử đọc file
    # Đọc kết quả từ file
    elo_result_file = os.path.join(DOVE_DIR, "current_elo.txt")
    current_elo = 1500  # Giá trị mặc định
    
    if os.path.exists(elo_result_file):
        try:
            with open(elo_result_file, "r") as f:
                current_elo = float(f.read().strip())
            log_message(f"Elo hiện tại: {current_elo}")
            # Coi như thành công nếu đọc được file
            return True, current_elo
        except Exception as e:
            log_message(f"Lỗi đọc file Elo: {str(e)}")
    
    if not success:
        log_message("Không tìm thấy file kết quả Elo, sử dụng giá trị mặc định 1500")
        # Tạo file với giá trị mặc định
        try:
            with open(elo_result_file, "w") as f:
                f.write("1500")
        except:
            pass
    
    return success, current_elo

def continue_training():
    """Tiếp tục huấn luyện sau khi đánh giá"""
    # Huấn luyện mạng đánh giá với dữ liệu mới
    train_cmd = f"cd {DOVE_DIR} && python train_evaluator.py"
    if not run_command(train_cmd, "Tiếp tục huấn luyện mạng đánh giá"):
        return False
    
    # Tiếp tục self-play
    gameplay_cmd = f"cd {DOVE_DIR} && python GamePlay.py"
    return run_command(gameplay_cmd, "Tiếp tục tự học qua Self-play")

def full_training_cycle(initial_depth=5, evaluation_depth=10):
    """Thực hiện toàn bộ chu trình huấn luyện"""
    log_message("===== BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN ĐẦY ĐỦ =====")
    
    # Kiểm tra xem có cần tạo mô hình ban đầu không
    eval_path = os.path.join(DOVE_DIR, "best_eval.pt")
    if not os.path.exists(eval_path):
        log_message("Mô hình đánh giá chưa tồn tại. Tạo mô hình ban đầu...")
        create_model_cmd = f"cd {DOVE_DIR} && python create_initial_model.py"
        if not run_command(create_model_cmd, "Tạo mô hình ban đầu"):
            log_message("Không thể tạo mô hình ban đầu. Đang tạo thủ công...")
            try:
                # Sử dụng EvaluatorNet đã import ở đầu file
                model = EvaluatorNet()
                torch.save(model.state_dict(), eval_path)
                log_message(f"Đã tạo mô hình ban đầu tại {eval_path}")
            except Exception as e:
                log_message(f"Lỗi khi tạo mô hình: {str(e)}")
                return False
    
    # 1. Cập nhật độ sâu ban đầu
    update_stockfish_depth(initial_depth)
    
    # 2. Học từ Stockfish trước
    log_message("GIAI ĐOẠN 1: HỌC TỪ STOCKFISH")
    if not train_from_stockfish():
        log_message("Giai đoạn 1 thất bại, dừng quy trình")
        return False
    
    # 3. Đánh giá Elo ban đầu
    log_message("GIAI ĐOẠN 2: ĐÁNH GIÁ ELO BAN ĐẦU")
    success, initial_elo = evaluate_elo(initial_depth)
    if success:
        log_message(f"Elo ban đầu: {initial_elo}")
    
    # 4. Chạy self-play
    log_message("GIAI ĐOẠN 3: TỰ HỌC QUA SELF-PLAY")
    if not run_self_play():
        log_message("Giai đoạn 3 thất bại, dừng quy trình")
        return False
    
    # 5. Đánh giá Elo sau self-play
    log_message("GIAI ĐOẠN 4: ĐÁNH GIÁ ELO SAU SELF-PLAY")
    success, after_selfplay_elo = evaluate_elo(initial_depth)
    if success:
        log_message(f"Elo sau self-play: {after_selfplay_elo}")
        if initial_elo > 0:
            log_message(f"Thay đổi Elo: {after_selfplay_elo - initial_elo:+.2f}")
    
    # 6. Đánh giá với Stockfish mức cao hơn
    log_message(f"GIAI ĐOẠN 5: THU THẬP DỮ LIỆU VỚI STOCKFISH ĐỘ SÂU {evaluation_depth}")
    update_stockfish_depth(evaluation_depth)
    stockfish_cmd = f"cd {DOVE_DIR} && python play_vs_stockfish.py"
    if not run_command(stockfish_cmd, f"Thu thập dữ liệu Stockfish độ sâu {evaluation_depth}"):
        log_message("Giai đoạn 5 thất bại, tiếp tục với dữ liệu hiện có")
    
    # 7. Tiếp tục huấn luyện
    log_message("GIAI ĐOẠN 6: HUẤN LUYỆN LẠI MẠNG ĐÁNH GIÁ")
    train_cmd = f"cd {DOVE_DIR} && python train_evaluator.py"
    if not run_command(train_cmd, "Huấn luyện lại mạng đánh giá"):
        log_message("Giai đoạn 6 thất bại, dừng quy trình")
        return False
    
    # 8. Tiếp tục self-play
    log_message("GIAI ĐOẠN 7: TIẾP TỤC TỰ HỌC QUA SELF-PLAY")
    if not run_self_play():
        log_message("Giai đoạn 7 thất bại, dừng quy trình")
        return False
    
    # 9. Đánh giá Elo cuối cùng
    log_message("GIAI ĐOẠN 8: ĐÁNH GIÁ ELO CUỐI CÙNG")
    success, final_elo = evaluate_elo(evaluation_depth)
    if success:
        log_message(f"Elo cuối cùng: {final_elo}")
        if after_selfplay_elo > 0:
            log_message(f"Thay đổi Elo so với sau self-play: {final_elo - after_selfplay_elo:+.2f}")
        if initial_elo > 0:
            log_message(f"Tổng thay đổi Elo: {final_elo - initial_elo:+.2f}")
    
    log_message("===== HOÀN THÀNH QUY TRÌNH HUẤN LUYỆN ĐẦY ĐỦ =====")
    return True

def iterative_training(target_elo=2200, max_iterations=10):
    """Lặp lại quy trình huấn luyện cho đến khi đạt được Elo mục tiêu"""
    log_message(f"===== BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN VÒNG LẶP (MỤC TIÊU: {target_elo}) =====")
    
    # Lưu lịch sử tiến bộ
    history = []
    current_elo = 0
    initial_depth = 5
    current_depth = initial_depth
    
    # Kiểm tra Elo hiện tại trước khi bắt đầu
    success, current_elo = evaluate_elo(current_depth)
    if not success:
        log_message("Không thể đánh giá Elo ban đầu, giả định 1500")
        current_elo = 1500
    
    history.append({
        "iteration": 0,
        "elo": current_elo,
        "stockfish_depth": current_depth
    })
    
    log_message(f"Elo ban đầu: {current_elo}")
    
    # Lặp lại quy trình cho đến khi đạt được Elo mục tiêu
    iteration = 1
    while current_elo < target_elo and iteration <= max_iterations:
        log_message(f"===== VÒNG LẶP {iteration}/{max_iterations} (Elo hiện tại: {current_elo}) =====")
        
        # Điều chỉnh độ sâu Stockfish theo Elo hiện tại
        if current_elo < 1600:
            current_depth = 5
        elif current_elo < 1800:
            current_depth = 8
        elif current_elo < 2000:
            current_depth = 10
        else:
            current_depth = 15
        
        log_message(f"Độ sâu Stockfish cho vòng lặp này: {current_depth}")
        
        # Chạy một chu trình huấn luyện đầy đủ
        full_training_cycle(current_depth, current_depth + 2)
        
        # Đánh giá lại sau khi hoàn thành chu trình
        success, new_elo = evaluate_elo(current_depth)
        if success:
            elo_gain = new_elo - current_elo
            current_elo = new_elo
            log_message(f"Elo sau vòng lặp {iteration}: {current_elo} (Thay đổi: {elo_gain:+.2f})")
            
            # Cập nhật lịch sử
            history.append({
                "iteration": iteration,
                "elo": current_elo,
                "stockfish_depth": current_depth,
                "elo_gain": elo_gain
            })
            
            # Lưu lịch sử
            with open("training_history.json", "w") as f:
                json.dump(history, f, indent=4)
        else:
            log_message("Không thể đánh giá Elo, tiếp tục với vòng lặp tiếp theo")
        
        # Nếu đạt được Elo mục tiêu, kết thúc
        if current_elo >= target_elo:
            log_message(f"Đã đạt được mục tiêu Elo: {current_elo} >= {target_elo}")
            break
        
        iteration += 1
    
    # Kết quả cuối cùng
    if current_elo >= target_elo:
        log_message(f"HUẤN LUYỆN THÀNH CÔNG! Elo cuối: {current_elo} sau {iteration} vòng lặp")
    else:
        log_message(f"ĐÃ ĐẠT SỐ VÒNG LẶP TỐI ĐA. Elo cuối: {current_elo}")
    
    # Tạo biểu đồ tiến trình
    plot_training_progress(history)
    
    return current_elo >= target_elo

def plot_training_progress(history):
    """Vẽ biểu đồ tiến trình huấn luyện"""
    try:
        import matplotlib.pyplot as plt
        
        iterations = [entry["iteration"] for entry in history]
        elos = [entry["elo"] for entry in history]
        depths = [entry["stockfish_depth"] for entry in history]
        
        # Tạo đồ thị 2 trục y
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Trục y bên trái: Elo
        ax1.set_xlabel('Vòng lặp')
        ax1.set_ylabel('Elo', color='tab:blue')
        ax1.plot(iterations, elos, 'o-', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Trục y bên phải: Độ sâu Stockfish
        ax2 = ax1.twinx()
        ax2.set_ylabel('Độ sâu Stockfish', color='tab:red')
        ax2.plot(iterations, depths, 's--', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Tiêu đề và lưới
        plt.title('Tiến trình huấn luyện AI cờ vua')
        plt.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Lưu đồ thị
        plt.savefig('training_progress.png')
        log_message("Đã tạo biểu đồ tiến trình huấn luyện: training_progress.png")
    except ImportError:
        log_message("Không thể tạo biểu đồ (cần cài đặt matplotlib)")

if __name__ == "__main__":
    # Tạo parser cho các đối số dòng lệnh
    parser = argparse.ArgumentParser(description="Huấn luyện AI cờ vua kết hợp học từ Stockfish và self-play")
    parser.add_argument("--mode", choices=["full", "stockfish", "selfplay", "evaluate", "continue", "iterative"],
                        default="full", help="Chế độ huấn luyện (mặc định: full)")
    parser.add_argument("--elo-only", action="store_true",
                        help="Chỉ đánh giá Elo và không huấn luyện")
    parser.add_argument("--target-elo", type=int, default=2200,
                        help="Mục tiêu Elo cho huấn luyện lặp lại (mặc định: 2200)")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Số vòng lặp tối đa cho huấn luyện lặp lại (mặc định: 10)")
    parser.add_argument("--initial-depth", type=int, default=5,
                        help="Độ sâu Stockfish ban đầu (mặc định: 5)")
    parser.add_argument("--evaluation-depth", type=int, default=10,
                        help="Độ sâu Stockfish để đánh giá (mặc định: 10)")
    
    args = parser.parse_args()
    
    # Tạo thư mục log nếu chưa tồn tại
    os.makedirs("logs", exist_ok=True)
    
    if args.elo_only:
        success, elo = evaluate_elo(args.evaluation_depth)
        if success:
            log_message(f"Elo hiện tại: {elo}")
    elif args.mode == "iterative":
        iterative_training(args.target_elo, args.max_iterations)
    elif args.mode == "full":
        full_training_cycle(args.initial_depth, args.evaluation_depth)
    elif args.mode == "stockfish":
        update_stockfish_depth(args.initial_depth)
        train_from_stockfish()
    elif args.mode == "selfplay":
        run_self_play()
    elif args.mode == "evaluate":
        success, elo = evaluate_elo(args.evaluation_depth)
        if success:
            log_message(f"Elo hiện tại: {elo}")
    elif args.mode == "continue":
        continue_training()