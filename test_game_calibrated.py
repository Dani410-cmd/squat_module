import sys
import time

import cv2
import pygame

from squat_control_module import SquatControlModule

WIDTH = 500
HEIGHT = 700
BIRD_X = 120
BIRD_SIZE = 40

SMOOTHING = 0.5
CALIBRATION_SAMPLES = 30
COUNTDOWN_SECONDS = 3
FPS = 60


def draw_center_text(frame, text_lines, color=(0, 255, 0)):
    if isinstance(text_lines, str):
        text_lines = [text_lines]

    h, w, _ = frame.shape

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    alpha = 0.55
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.6
    thickness = 3
    line_gap = 20

    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in text_lines]
    total_h = sum(s[1] for s in sizes) + line_gap * (len(text_lines) - 1)
    start_y = (h - total_h) // 2

    y = start_y
    for line, (tw, th) in zip(text_lines, sizes):
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
        remaining = int(seconds - elapsed) + 1

        if elapsed >= seconds:
            break

        draw_center_text(frame, [message, str(remaining)], (0, 255, 255))
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            return False

    return True


def show_status(detector, message, color, duration=2):
    start = time.time()

    while time.time() - start < duration:
        frame, _, _ = detector.read_frame()
        if frame is None:
            continue

        draw_center_text(frame, [message], color)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            return False

    return True


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Squat Controlled Bird")
    clock = pygame.time.Clock()

    detector = None

    try:
        detector = SquatControlModule()
    except RuntimeError as e:
        print(e)
        pygame.quit()
        sys.exit()

    smoothed_ratio = 0.0

    try:
        if not countdown(detector, COUNTDOWN_SECONDS, "STAND STRAIGHT"):
            return

        if not detector.calibrate_standing(CALIBRATION_SAMPLES):
            return

        if not countdown(detector, COUNTDOWN_SECONDS, "GO DOWN"):
            return

        if not detector.calibrate_squat(CALIBRATION_SAMPLES):
            return

        if not detector.is_calibrated():
            show_status(detector, "CALIBRATION FAILED", (0, 0, 255))
            return

        if not show_status(detector, "READY", (0, 255, 0)):
            return

        running = True
        while running:
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            frame, ratio = detector.get_ratio()

            target_ratio = smoothed_ratio if ratio is None else ratio
            smoothed_ratio = (1 - SMOOTHING) * smoothed_ratio + SMOOTHING * target_ratio

            max_y = HEIGHT - BIRD_SIZE
            bird_y = smoothed_ratio * max_y

            screen.fill((135, 206, 235))

            pygame.draw.rect(
                screen,
                (255, 255, 0),
                (BIRD_X, int(bird_y), BIRD_SIZE, BIRD_SIZE),
            )

            pygame.display.flip()

            if frame is not None:
                cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                running = False

    finally:
        if detector is not None:
            detector.release()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
