import cv2
import threading
from config import Config
from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer

class ImageProcessor:
    def __init__(self, face_database):
        self.face_database = face_database
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.last_face_encodings = {}

        self.pause_face_recognition = False
        self.lock = threading.Lock()

    def pause_recognition(self):
        """Приостанавливает распознавание лиц."""
        with self.lock:
            self.pause_face_recognition = True

    def resume_recognition(self):
        """Возобновляет распознавание лиц."""
        with self.lock:
            self.pause_face_recognition = False

    def convert_to_grayscale(self, frame):
        """Преобразует кадр в оттенки серого."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, frame, kernel_size = (5, 5), sigmaX=0):
        """Применяет размытие по Гауссу к кадру."""
        return cv2.GaussianBlur(frame, kernel_size, sigmaX)
   
    def process_frame(self, frame):
        """Выполняет предварительную обработку и распознавание лиц."""
        with self.lock:
            if self.pause_face_recognition:
                return frame

            processed_frame = frame
            if 'grayscale' in Config().IMAGE_PROCESSORS:
                processed_frame = self.convert_to_grayscale(processed_frame)
            if 'blur' in Config().IMAGE_PROCESSORS:
                processed_frame = self.gaussian_blur(processed_frame)

            if 'face_detect' in Config().IMAGE_PROCESSORS:
                faces = self.face_detector.detect_faces(processed_frame)

                for (left, top, right, bottom) in faces:
                    x, y, w, h = left, top, right-left, bottom-top

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face_image = frame[y:y+h, x:x+w]

                    display_unknown_label = True
                    face_encodings = self.face_recognizer.get_face_encodings(face_image)

                    if face_encodings:
                        known_face_encodings = self.face_database.get_face_encodings()

                        # Получаем список булевых значений вместо одного
                        matches = self.face_recognizer.compare_faces(
                            face_encodings, 
                            known_face_encodings
                        )

                        # Находим все совпадения
                        matched_indices = [i for i, match in enumerate(matches) if match]
                        if matched_indices:
                            matched_id = self.face_database.get_face_ids()[matched_indices[0]]

                            if matched_id:
                                display_unknown_label = False
                                cv2.putText(frame, matched_id, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    if display_unknown_label:
                        cv2.putText(frame, 'Uncknown', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            return frame
