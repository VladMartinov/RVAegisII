import cv2
import time
import numpy as np
import threading
from config import Config
from core.face_database import FaceDatabase
from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.image_processor import ImageProcessor
from core.camera import CameraManager

class FaceRecognitionAI:
    def __init__(self):
        # Инициализация компонентов
        self.face_database = FaceDatabase()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.image_processor = ImageProcessor(self.face_database)
        self.camera_manager = CameraManager(self.image_processor)

        # Переменная для хранения текущего кадра
        self.frames = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        if Config().SHOW_CAMERA_WINDOW:
            # Запускаем поток для отображения изображений
            self.display_thread = threading.Thread(target=self.display_frames)
            self.display_thread.start()

    def display_frames(self):
        """Поток для отображения изображений."""
        while not self.stop_event.is_set():
            with self.lock:
                for camera_index, all_frames in self.frames.items():
                    for frame in all_frames:
                        cv2.imshow(f"Camera {camera_index}", frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.stop_event.set()
                            break

    def add_images(self, images, labels):
        """
        Добавляет изображения и лейблы в базу данных.
        :param images: Список изображений в формате bytes.
        :param labels: Список лейблов для изображений.
        """

        # Приостанавливаем распознавание лиц
        self.image_processor.pause_recognition()

        # Очищаем базу данных перед добавлением новых данных
        self.face_database.clear()

        for image_bytes, label in zip(images, labels):
            # Преобразуем bytes в изображение
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_encodings = self.face_recognizer.get_face_encodings(image_rgb)
            if face_encodings:
                self.face_database.add_face(label, face_encodings[0])
                print(f"[INFO] Added face with label: {label}")
            else:
                print(f"[WARNING] No faces found in image for label: {label}")

        # Возобновляем распознавание лиц
        self.image_processor.resume_recognition()

    def start_camera_processing(self):
        """
        Запускает обработку изображений с камеры.
        """
        self.camera_manager.start_capture()

        while not self.stop_event.is_set():
            frames = self.camera_manager.get_frames()
            with self.lock:
                self.frames = frames
            time.sleep(Config().FPS_RETURNING / 100)

        # Останавливаем захват кадров
        self.camera_manager.stop_capture()
        if Config().SHOW_CAMERA_WINDOW:
            cv2.destroyAllWindows()

    def get_current_frames(self):
        """
        Возвращает текущие кадры с камер.
        :return: Словарь, где ключи - индексы камер, значения - списки кадров (numpy arrays).
        """
        with self.lock:
            return self.frames.copy()

    def stop(self):
        """Останавливает поток отображения."""
        self.stop_event.set()
        if Config().SHOW_CAMERA_WINDOW:
            self.display_thread.join()