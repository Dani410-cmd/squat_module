import random
import sys
import time

import cv2
import pygame

from squat_control_module import SquatControlModule


SCREEN_WIDTH = 500
SCREEN_HEIGHT = 700
FPS = 60

BIRD_X = 120
BIRD_SIZE = 34

GROUND_HEIGHT = 90

PIPE_WIDTH = 65
PIPE_GAP = 260
PIPE_SPEED = 6
PIPE_SPAWN_MS = 1800


CALIBRATION_SAMPLES = 30
COUNTDOWN_SECONDS = 3

# 1.0 = вообще без сглаживания
# 0.85 = почти прямой контроль
# 0.7 = чуть плавнее
POSITION_SMOOTHING = 0.5


SKY_COLOR = (135, 206, 235)
GROUND_COLOR = (222, 184, 135)
PIPE_COLOR = (0, 180, 0)
PIPE_DARK = (0, 130, 0)
BIRD_COLOR = (255, 230, 0)
TEXT_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)


def draw_text(screen, text, size, x, y, center=False, color=TEXT_COLOR):
    font = pygame.font.SysFont("arial", size, bold=True)
    shadow = font.render(text, True, SHADOW_COLOR)
    surf = font.render(text, True, color)

    rect = surf.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)

    shadow_rect = rect.copy()
    shadow_rect.x += 2
    shadow_rect.y += 2

    screen.blit(shadow, shadow_rect)
    screen.blit(surf, rect)


def draw_bird(screen, x, y, size):
    rect = pygame.Rect(int(x), int(y), size, size)
    pygame.draw.ellipse(screen, BIRD_COLOR, rect)
    eye_x = rect.x + int(size * 0.68)
    eye_y = rect.y + int(size * 0.32)
    pygame.draw.circle(screen, (255, 255, 255), (eye_x, eye_y), max(2, size // 8))
    pygame.draw.circle(screen, (0, 0, 0), (eye_x, eye_y), max(1, size // 16))

    beak = [
        (rect.x + int(size * 0.92), rect.y + int(size * 0.48)),
        (rect.x + int(size * 1.12), rect.y + int(size * 0.40)),
        (rect.x + int(size * 0.92), rect.y + int(size * 0.62)),
    ]
    pygame.draw.polygon(screen, (255, 140, 0), beak)


def draw_pipe(screen, pipe):
    x = pipe["x"]
    gap_y = pipe["gap_y"]

    top_rect = pygame.Rect(x, 0, PIPE_WIDTH, gap_y)
    bottom_rect = pygame.Rect(
        x,
        gap_y + PIPE_GAP,
        PIPE_WIDTH,
        SCREEN_HEIGHT - GROUND_HEIGHT - (gap_y + PIPE_GAP),
    )

    pygame.draw.rect(screen, PIPE_COLOR, top_rect)
    pygame.draw.rect(screen, PIPE_COLOR, bottom_rect)

    pygame.draw.rect(screen, PIPE_DARK, top_rect, 4)
    pygame.draw.rect(screen, PIPE_DARK, bottom_rect, 4)

    cap_h = 20
    top_cap = pygame.Rect(x - 5, gap_y - cap_h, PIPE_WIDTH + 10, cap_h)
    bottom_cap = pygame.Rect(x - 5, gap_y + PIPE_GAP, PIPE_WIDTH + 10, cap_h)

    pygame.draw.rect(screen, PIPE_COLOR, top_cap)
    pygame.draw.rect(screen, PIPE_COLOR, bottom_cap)
    pygame.draw.rect(screen, PIPE_DARK, top_cap, 4)
    pygame.draw.rect(screen, PIPE_DARK, bottom_cap, 4)


def create_pipe():
    min_gap_y = 140
    max_gap_y = SCREEN_HEIGHT - GROUND_HEIGHT - PIPE_GAP - 140
    return {
        "x": SCREEN_WIDTH + 40,
        "gap_y": random.randint(min_gap_y, max_gap_y),
        "passed": False,
    }


def get_bird_rect(y):
    return pygame.Rect(BIRD_X, int(y), BIRD_SIZE, BIRD_SIZE)


def check_collision(bird_y, pipes):
    bird_rect = get_bird_rect(bird_y)

    if bird_rect.top < 0:
        return True

    if bird_rect.bottom > SCREEN_HEIGHT - GROUND_HEIGHT:
        return True

    for pipe in pipes:
        top_rect = pygame.Rect(pipe["x"], 0, PIPE_WIDTH, pipe["gap_y"])
        bottom_rect = pygame.Rect(
            pipe["x"],
            pipe["gap_y"] + PIPE_GAP,
            PIPE_WIDTH,
            SCREEN_HEIGHT - GROUND_HEIGHT - (pipe["gap_y"] + PIPE_GAP),
        )

        if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
            return True

    return False


def draw_camera_center_text(frame, lines, color=(0, 255, 0)):
    if isinstance(lines, str):
        lines = [lines]

    h, w, _ = frame.shape

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    alpha = 0.55
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thickness = 3
    line_gap = 20

    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    total_h = sum(s[1] for s in sizes) + line_gap * (len(lines) - 1)
    start_y = (h - total_h) // 2

    y = start_y
    for line, (tw, th) in zip(lines, sizes):
        x = (w - tw) // 2
        cv2.putText(
            frame, line, (x, y + th), font, scale, color, thickness, cv2.LINE_AA
        )
        y += th + line_gap


def countdown(detector, seconds, message):
    start = time.time()

    while True:
        frame, _, _ = detector.read_frame()
        if frame is None:
            continue

        elapsed = time.time() - start
        if elapsed >= seconds:
            break

        remaining = int(seconds - elapsed) + 1
        draw_camera_center_text(frame, [message, str(remaining)], (0, 255, 255))
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:
            return False

    return True


def show_camera_status(detector, message, color=(0, 255, 0), duration=1.5):
    start = time.time()

    while time.time() - start < duration:
        frame, _, _ = detector.read_frame()
        if frame is None:
            continue

        draw_camera_center_text(frame, [message], color)
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:
            return False

    return True


def calibrate(detector):
    if not countdown(detector, COUNTDOWN_SECONDS, "STAND STRAIGHT"):
        return False

    if not detector.calibrate_standing(CALIBRATION_SAMPLES):
        return False

    if not countdown(detector, COUNTDOWN_SECONDS, "GO DOWN"):
        return False

    if not detector.calibrate_squat(CALIBRATION_SAMPLES):
        return False

    if not detector.is_calibrated():
        show_camera_status(detector, "CALIBRATION FAILED", (0, 0, 255), 2.0)
        return False

    if not show_camera_status(detector, "READY", (0, 255, 0), 1.2):
        return False

    return True


def draw_game(screen, bird_y, pipes, score, game_over):
    screen.fill(SKY_COLOR)

    for pipe in pipes:
        draw_pipe(screen, pipe)

    ground_rect = pygame.Rect(
        0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT
    )
    pygame.draw.rect(screen, GROUND_COLOR, ground_rect)

    draw_bird(screen, BIRD_X, bird_y, BIRD_SIZE)

    draw_text(screen, f"Score: {score}", 34, 18, 18)

    if game_over:
        draw_text(
            screen,
            "GAME OVER",
            54,
            SCREEN_WIDTH // 2,
            SCREEN_HEIGHT // 2 - 40,
            center=True,
        )
        draw_text(
            screen,
            "Press R to restart",
            28,
            SCREEN_WIDTH // 2,
            SCREEN_HEIGHT // 2 + 20,
            center=True,
        )
        draw_text(
            screen,
            "Press C to recalibrate",
            24,
            SCREEN_WIDTH // 2,
            SCREEN_HEIGHT // 2 + 58,
            center=True,
        )

    pygame.display.flip()


def reset_round():
    pipes = [create_pipe()]
    score = 0
    game_over = False
    spawn_timer = 0

    max_y = SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_SIZE
    bird_y = max_y * 0.4
    smoothed_ratio = 0.4

    return pipes, score, game_over, spawn_timer, bird_y, smoothed_ratio


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird Squat Control")
    clock = pygame.time.Clock()

    try:
        detector = SquatControlModule()
    except RuntimeError as e:
        print(e)
        pygame.quit()
        sys.exit(1)

    try:
        if not calibrate(detector):
            detector.release()
            pygame.quit()
            sys.exit()

        pipes, score, game_over, spawn_timer, bird_y, smoothed_ratio = reset_round()

        running = True
        while running:
            dt = clock.tick(FPS)
            spawn_timer += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if game_over and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        pipes, score, game_over, spawn_timer, bird_y, smoothed_ratio = (
                            reset_round()
                        )
                    elif event.key == pygame.K_c:
                        if calibrate(detector):
                            (
                                pipes,
                                score,
                                game_over,
                                spawn_timer,
                                bird_y,
                                smoothed_ratio,
                            ) = reset_round()

            frame, ratio = detector.get_ratio()

            if not game_over:
                if ratio is not None:
                    smoothed_ratio = (
                        1 - POSITION_SMOOTHING
                    ) * smoothed_ratio + POSITION_SMOOTHING * ratio

                max_y = SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_SIZE
                bird_y = smoothed_ratio * max_y

                if spawn_timer >= PIPE_SPAWN_MS:
                    pipes.append(create_pipe())
                    spawn_timer = 0

                for pipe in pipes:
                    pipe["x"] -= PIPE_SPEED

                    if not pipe["passed"] and pipe["x"] + PIPE_WIDTH < BIRD_X:
                        pipe["passed"] = True
                        score += 1

                pipes = [pipe for pipe in pipes if pipe["x"] + PIPE_WIDTH > -20]

                if check_collision(bird_y, pipes):
                    game_over = True

            draw_game(screen, bird_y, pipes, score, game_over)

            if frame is not None:
                cv2.imshow("Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:
                running = False

    finally:
        detector.release()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
