import cv2
import mediapipe as mp


class SquatControlModule:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру")

        self.standing_value = None
        self.squat_value = None

    def _avg_y(self, landmarks, left_landmark, right_landmark):
        left_point = landmarks[left_landmark.value]
        right_point = landmarks[right_landmark.value]
        return (left_point.y + right_point.y) / 2.0

    def _clamp01(self, value):
        return max(0.0, min(1.0, value))

    def get_hip_y(self, landmarks):
        return self._avg_y(
            landmarks,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
        )

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return frame, None, None

        landmarks = results.pose_landmarks.landmark
        current_value = self.get_hip_y(landmarks)

        return frame, results, current_value

    def calibrate_standing(self, samples=30):
        values = []

        while len(values) < samples:
            frame, _, current_value = self.read_frame()
            if frame is None:
                continue

            if current_value is not None:
                values.append(current_value)

            cv2.putText(
                frame,
                "CALIBRATION: STAND STILL",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Samples: {len(values)}/{samples}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                return False

        self.standing_value = min(values)
        return True

    def calibrate_squat(self, samples=30):
        values = []

        while len(values) < samples:
            frame, _, current_value = self.read_frame()
            if frame is None:
                continue

            if current_value is not None:
                values.append(current_value)

            cv2.putText(
                frame,
                "CALIBRATION: GO TO LOWEST SQUAT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Samples: {len(values)}/{samples}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                return False

        self.squat_value = max(values)
        return True

    def is_calibrated(self):
        return (
            self.standing_value is not None
            and self.squat_value is not None
            and abs(self.squat_value - self.standing_value) > 1e-6
        )

    def get_ratio(self):
        frame, _, current_value = self.read_frame()

        if frame is None:
            return None, None

        if not self.is_calibrated() or current_value is None:
            return frame, None

        denominator = self.squat_value - self.standing_value
        if abs(denominator) < 1e-6:
            return frame, None

        ratio = (current_value - self.standing_value) / denominator
        ratio = self._clamp01(ratio)

        return frame, ratio

    def release(self):
        self.cap.release()
        self.pose.close()
        cv2.destroyAllWindows()
