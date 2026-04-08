import cv2
import mediapipe as mp


class SquatControlModule:
    def __init__(self, camera_index=0, camera_backend=None):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.cap = self._open_camera(camera_index, camera_backend)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(
                "Не удалось открыть камеру. Проверь доступ к камере и camera_index."
            )

        self.standing_hip_y = None
        self.squat_hip_y = None

    def _open_camera(self, camera_index, camera_backend):
        backends = []

        if camera_backend is not None:
            backends.append(camera_backend)

        if hasattr(cv2, "CAP_AVFOUNDATION"):
            backends.append(cv2.CAP_AVFOUNDATION)
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)

        backends.append(None)

        for backend in backends:
            cap = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
            if cap is not None and cap.isOpened():
                return cap

            if cap is not None:
                cap.release()

        return None

    def get_current_hip_y(self, landmarks):
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        return (left_hip.y + right_hip.y) / 2

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
        hip_y = self.get_current_hip_y(landmarks)

        return frame, results, hip_y

    def is_calibrated(self):
        return (
            self.standing_hip_y is not None
            and self.squat_hip_y is not None
            and self.squat_hip_y > self.standing_hip_y
        )

    def get_ratio(self):
        frame, results, hip_y = self.read_frame()

        if frame is None:
            return None, None

        if not self.is_calibrated() or hip_y is None:
            return frame, None

        denominator = self.squat_hip_y - self.standing_hip_y
        if denominator <= 0:
            return frame, None

        ratio = (hip_y - self.standing_hip_y) / denominator
        ratio = max(0.0, min(1.0, ratio))

        cv2.putText(
            frame,
            f"Ratio: {ratio:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return frame, ratio

    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.pose.close()
        cv2.destroyAllWindows()
