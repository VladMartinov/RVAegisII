import cv2
import os
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from core.frame_queue import FrameQueue
from core.image_processor import ImageProcessor
from config import Config

class CameraManager:
    def __init__(self, face_database):
        self.cameras = []
        self.lock = Lock()
        self.is_running = False
        self.camera_threads = {}
        self.frame_queues = {}
        self.image_processor = ImageProcessor(face_database)

        max_workers = min(Config().NUM_CAMERAS, os.cpu_count() or 1)
        self.thread_pool = ThreadPoolExecutor(max_workers = max_workers)

    def get_available_cameras(self):
        """Находит доступные камеры."""
        available_cameras = []
        for i in range(Config().NUM_CAMERAS):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def start_capture(self):
        """Запускает захват кадров с камер в отдельных потоках."""
        if self.is_running:
            return

        self.is_running = True
        self.cameras = self.get_available_cameras()

        for camera_index in self.cameras:
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config().CAMERA_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config().CAMERA_RESOLUTION[1])
            cap.set(cv2.CAP_PROP_FPS, Config().CAMERA_FPS)

            if not cap.isOpened():
                print(f"Не удалось открыть камеру с индексом {camera_index}")
                continue

            frame_queue = FrameQueue(max_size = Config().MAX_FRAMES_IN_QUEUE)
            self.thread_pool.submit(self._capture_loop, camera_index, cap, frame_queue)  # Используем пул потоков

            self.camera_threads[camera_index] = cap
            self.frame_queues[camera_index] = frame_queue

    def _capture_loop(self, camera_index, cap, frame_queue):
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print(f"Не удалось получить изображение с камеры {camera_index}")
                break
            processed_frame = self.image_processor.process_frame(frame)     # Обрабатываем кадр
            frame_queue.put(processed_frame)                                # Сохраняем обработанный кадр

        with self.lock:
            if camera_index in self.camera_threads:
                self.camera_threads.pop(camera_index)
            if camera_index in self.frame_queues:
                self.frame_queues.pop(camera_index)

        cap.release()

    def get_frames(self):
        """Возвращает все кадры из очереди."""
        frames = {}
        for camera_index, frame_queue in self.frame_queues.items():
            frames[camera_index] = frame_queue.get_all()
        return frames

    def stop_capture(self):
        """Останавливает захват кадров и освобождает ресурсы."""
        self.is_running = False

        # Создаем копию ключей для итерации
        self.thread_pool.shutdown(wait=True)  # Ожидаем завершения всех потоков

        # Очищаем словари, так как все потоки завершены
        self.cameras = []
        self.camera_threads = {}
        self.frame_queues = {}

    def __del__(self):
        self.stop_capture()