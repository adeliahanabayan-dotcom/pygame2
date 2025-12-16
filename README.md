# pygame2
import cv2
import numpy as np
import pygame
import random

# =========================
# PYGAME SETUP
# =========================
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Motion Control Game (No MediaPipe)")
clock = pygame.time.Clock()
FPS = 30

WHITE = (255, 255, 255)
RED   = (255, 50, 50)
BLUE  = (50, 100, 255)
BLACK = (0, 0, 0)

# Paddle
PADDLE_W = 150
PADDLE_H = 20
paddle_x = WIDTH // 2 - PADDLE_W // 2
paddle_y = HEIGHT - 60

# Ball
BALL_R = 15
ball_x = random.randint(50, WIDTH - 50)
ball_y = -20
ball_speed = 5

score = 0
lives = 3

font = pygame.font.SysFont(None, 32)
def draw_text(t, x, y, col=WHITE):
    screen.blit(font.render(t, True, col), (x, y))

# =========================
# OPENCV MOTION DETECTION
# =========================
cap = cv2.VideoCapture(0)

# background subtractor (deteksi gerakan)
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)


# =========================
# MAIN GAME LOOP
# =========================
running = True
while running:

    # ---- EVENT PYGAME ----
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ---- READ CAMERA ----
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membuka kamera.")
        break

    frame = cv2.flip(frame, 1)  # mirror
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = fgbg.apply(gray)

    # noise removal
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # cari kontur area gerakan
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Ambil kontur terbesar (pergerakan utama)
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1500:  # threshold gerakan
            x, y, w, h = cv2.boundingRect(largest)
            center_x = x + w // 2

            # Mapping dari ukuran kamera ke layar game
            paddle_x = int(center_x / frame.shape[1] * (WIDTH - PADDLE_W))

    # ---- UPDATE BALL ----
    ball_y += ball_speed

    # tabrak paddle
    if paddle_y - BALL_R < ball_y < paddle_y + BALL_R:
        if paddle_x < ball_x < paddle_x + PADDLE_W:
            score += 1
            ball_speed += 0.3
            ball_y = -20
            ball_x = random.randint(50, WIDTH - 50)

    # bolanya jatuh → nyawa berkurang
    if ball_y > HEIGHT:
        lives -= 1
        ball_y = -20
        ball_x = random.randint(50, WIDTH - 50)
        ball_speed = max(4, ball_speed - 0.3)

        if lives <= 0:
            screen.fill(BLACK)
            draw_text("GAME OVER", WIDTH//2 - 100, HEIGHT//2 - 40, RED)
            draw_text(f"Score: {score}", WIDTH//2 - 70, HEIGHT//2 + 10)
            draw_text("Tekan R untuk main lagi atau Q untuk keluar", WIDTH//2 - 230, HEIGHT//2 + 50)
            pygame.display.update()

            waiting = True
            while waiting:
                for e in pygame.event.get():
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_q:
                            running = False
                            waiting = False
                        if e.key == pygame.K_r:
                            score = 0
                            lives = 3
                            ball_speed = 5
                            waiting = False

    # ---- GAMBAR SEMUA ----
    screen.fill(BLACK)

    pygame.draw.rect(screen, BLUE, (paddle_x, paddle_y, PADDLE_W, PADDLE_H))
    pygame.draw.circle(screen, RED, (ball_x, ball_y), BALL_R)

    draw_text(f"Score: {score}", 10, 10)
    draw_text(f"Lives: {lives}", 10, 40)

    # Tampilkan preview kamera kecil
    preview = cv2.resize(frame, (200, 150))
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview_surface = pygame.surfarray.make_surface(np.rot90(preview))
    screen.blit(preview_surface, (WIDTH - 220, 10))

    pygame.display.update()
    clock.tick(FPS)


cap.release()
pygame.quit()
