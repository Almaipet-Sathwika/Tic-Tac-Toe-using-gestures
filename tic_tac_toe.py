import cv2
import numpy as np
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

board = [["" for _ in range(3)] for _ in range(3)]

def draw_board(img, start_x, start_y, size):
    cell_size = size // 3
    color_line = (255, 255, 255)
    thickness = 3
    for i in range(1, 3):
        cv2.line(img, (start_x, start_y + i * cell_size), (start_x + size, start_y + i * cell_size), color_line, thickness)
        cv2.line(img, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + size), color_line, thickness)

    for y in range(3):
        for x in range(3):
            center = (start_x + x * cell_size + cell_size // 2, start_y + y * cell_size + cell_size // 2)
            if board[y][x] == "X":
                offset = 40
                cv2.line(img, (center[0] - offset, center[1] - offset), (center[0] + offset, center[1] + offset), (0, 0, 255), 5)
                cv2.line(img, (center[0] + offset, center[1] - offset), (center[0] - offset, center[1] + offset), (0, 0, 255), 5)
            elif board[y][x] == "O":
                cv2.circle(img, center, 40, (0, 255, 0), 5)

def get_cell_from_position(x, y, start_x, start_y, size):
    if not (start_x <= x < start_x + size and start_y <= y < start_y + size):
        return None, None
    cell_size = size // 3
    cx = (x - start_x) // cell_size
    cy = (y - start_y) // cell_size
    return int(cx), int(cy)

def check_winner():
    for r in board:
        if r[0] == r[1] == r[2] != "":
            print("Winner found in row:", r[0])
            return r[0]
    for c in range(3):
        if board[0][c] == board[1][c] == board[2][c] != "":
            print("Winner found in column:", board[0][c])
            return board[0][c]
    if board[0][0] == board[1][1] == board[2][2] != "":
        print("Winner found in diagonal:", board[0][0])
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != "":
        print("Winner found in anti-diagonal:", board[0][2])
        return board[0][2]
    if all(board[r][c] != "" for r in range(3) for c in range(3)):
        print("Game draw")
        return "Draw"
    return None

def computer_move():
    empty = [(r, c) for r in range(3) for c in range(3) if board[r][c] == ""]
    if empty:
        r, c = random.choice(empty)
        board[r][c] = "O"

STATE_YOUR_TURN = 0
STATE_COMPUTER_TURN = 1
STATE_GAME_OVER = 2

cap = cv2.VideoCapture(0)
window_width, window_height = 600, 680  # Increased height to fit text below board
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

start_x, start_y = 10, 10
board_size = 580

game_state = STATE_YOUR_TURN
drawn = False
winner = None
last_computer_move_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (window_width, window_height))
    h, w = img.shape[:2]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if game_state == STATE_YOUR_TURN:
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            x, y = int(lm[8].x * w), int(lm[8].y * h)
            cv2.circle(img, (x, y), 12, (255, 0, 0), cv2.FILLED)

            if not drawn:
                cx, cy = get_cell_from_position(x, y, start_x, start_y, board_size)
                if cx is not None and 0 <= cx < 3 and 0 <= cy < 3 and board[cy][cx] == "":
                    board[cy][cx] = "X"
                    drawn = True

                    winner = check_winner()
                    if winner:
                        game_state = STATE_GAME_OVER
                    else:
                        game_state = STATE_COMPUTER_TURN
                        last_computer_move_time = time.time()
        else:
            drawn = False

    elif game_state == STATE_COMPUTER_TURN:
        if time.time() - last_computer_move_time > 0.7:
            computer_move()
            winner = check_winner()
            if winner:
                game_state = STATE_GAME_OVER
            else:
                game_state = STATE_YOUR_TURN
            drawn = False

    draw_board(img, start_x, start_y, board_size)

    font = cv2.FONT_HERSHEY_SIMPLEX
    status_text = ""
    if game_state == STATE_GAME_OVER:
        if winner == "Draw":
            status_text = "Game Over: Draw!"
        else:
            status_text = f"Game Over: {winner} Wins!"
    elif game_state == STATE_YOUR_TURN:
        status_text = "Your Turn"
    else:
        status_text = "Computer's Turn"

    # Draw status text clearly below board
    text_pos_y = start_y + board_size + 50
    cv2.putText(img, status_text, (start_x, text_pos_y), font, 1, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "Press 'r' to replay | 'q' to quit", (10, window_height - 20), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Tic Tac Toe - Draw X by hovering finger", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        board = [["" for _ in range(3)] for _ in range(3)]
        game_state = STATE_YOUR_TURN
        drawn = False
        winner = None

cap.release()
cv2.destroyAllWindows()
