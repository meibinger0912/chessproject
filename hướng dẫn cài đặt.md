D:/Trí tuệ nhân tạo/chess project/
Chess project/
├── main.py      # Điểm vào chính, quản lý menu và điều hướng
├── graphic.py  # Xử lý độ họa chung và các hàm vẽ
├── gui.py         # Giao diện khi người chơi đấu với Stockfish
├── guix.py       # Giao diện khi người chơi đấu với module AI 
├── chess_logic.py  # Logic cờ vua
├── chess_clock.py  # Đồng hồ đếm thời gian
├── assets/         # Thư mục chứa ảnh ô bàn cờ, quân cờ,…
│   ├── board.png
│   ├── white_pawn.png
│   ├── black_pawn.png
│   └── ... (other chess piece images)
├── stockfish/         # Stockfish chess engine download trên mạng 
│   └── stockfish.exe
└── ChessDove/     # Thư mục chứa mã nguồn AI tự phát triển  
    ├── ChessDove.py          # Module AI chính
    ├── ComputingAI.py      # Đánh giá vị trí các quân cờ
    ├── play_vs_stockfish.py   # Đấu cùng stockfish để lấy dữ liệu huấn luyện 
    ├── train_evaluator.py    # Huấn luyện mạng neural
    ├── evaluator.py             # Neural network
    ├── GamePlay.py           # Self-play và học tăng cường
    ├── watch_ai_battle.py  # Giao diện xem AI đấu với nhau
    └── evaluate_elo.py       # Đánh giá điểm Elo của AI

link tải AI:
    tải stockfish: https://stockfishchess.org/download/
    tải chessdove: https://drive.google.com/drive/folders/1Ih4kb9neKxhtTZBh43zYhwY_iOJg8lkF?usp=drive_link
    file drive toàn bộ chương trình:
    https://drive.google.com/drive/folders/1IOJsiKA1regz-Ze3X8S-6K6cXum3ZwAf?usp=sharing
