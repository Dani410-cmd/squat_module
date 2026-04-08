import sys
import time

import cv2
import pygame

from squat_control_module import SquatControlModule

WIDTH = 500
HEIGHT = 700
BIRD_X = 120
BIRD_SIZE = 40
SMOOTHING = 0.15
CALIBRATION_SAMPLES = 90
COUNTDOWN_SECONDS = 4
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
    scale = 1.8
    thickness = 4
    line_gap = 24

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


def calibrate_position(detector, samples, message):
    values = []

    while len(values) < samples:
        frame, _, hip_y = detector.read_frame()
        if frame is None:
            continue

        if hip_y is not None:
            values.append(hip_y)

        draw_center_text(frame, [message, f"{len(values)} / {samples}"])
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            return None

    return sum(values) / len(values) if values else None


def show_status(detector, message, color, duration_seconds=2):
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
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

    try:
        detector = SquatControlModule()
    except RuntimeError as error:
        print(error)
        pygame.quit()
        sys.exit(1)

    smoothed_ratio = 0.0

    try:
        if not countdown(detector, COUNTDOWN_SECONDS, "STAND STRAIGHT"):
            return

        detector.standing_hip_y = calibrate_position(
            detector, CALIBRATION_SAMPLES, "CALIBRATING STAND"
        )
        if detector.standing_hip_y is None:
            return

        if not countdown(detector, COUNTDOWN_SECONDS, "GO TO LOWEST SQUAT"):
            return

        detector.squat_hip_y = calibrate_position(
            detector, CALIBRATION_SAMPLES, "CALIBRATING SQUAT"
        )
        if detector.squat_hip_y is None:
            return

        if not detector.is_calibrated():
            show_status(detector, "CALIBRATION FAILED", (0, 0, 255), 2)
            return

        if not show_status(detector, "CALIBRATION DONE", (0, 255, 0), 2):
            return

        running = True
        while running:
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            camera_frame, ratio = detector.get_ratio()

            screen.fill((135, 206, 235))

            target_ratio = 0.0 if ratio is None else ratio
            smoothed_ratio = (1 - SMOOTHING) * smoothed_ratio + SMOOTHING * target_ratio

            max_y = HEIGHT - BIRD_SIZE
            bird_y = smoothed_ratio * max_y

            pygame.draw.rect(
                screen, (255, 255, 0), (BIRD_X, bird_y, BIRD_SIZE, BIRD_SIZE)
            )
            pygame.display.flip()

            if camera_frame is not None:
                cv2.putText(
                    camera_frame,
                    "CONTROL WITH SQUAT",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 255, 0),
                    3,
                )

                if ratio is not None:
                    cv2.putText(
                        camera_frame,
                        f"RATIO: {ratio:.2f}",
                        (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        3,
                    )
                else:
                    cv2.putText(
                        camera_frame,
                        "NO POSE",
                        (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )

                cv2.imshow("Camera", camera_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:
                running = False
    finally:
        detector.release()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
