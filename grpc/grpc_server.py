import grpc
from concurrent import futures
import face_recognition_pb2
import face_recognition_pb2_grpc
import sys
import os
import cv2
import time
import threading

# Получаем абсолютный путь к папке ai
ai_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../ai'))
sys.path.append(ai_path)

from face_recognition_ai import FaceRecognitionAI

class FaceRecognitionServicer(face_recognition_pb2_grpc.FaceRecognitionServicer):
    def __init__(self):
        # Инициализация ИИ
        self.face_recognition_ai = FaceRecognitionAI()

        # Запускаем обработку изображений с камеры в отдельном потоке
        self.camera_thread = threading.Thread(target = self.face_recognition_ai.start_camera_processing)
        self.camera_thread.start()

    def SendImages(self, request, context):
        print("[INFO] Received images and labels from C# gRPC server.")
        print(f"[INFO] Received {len(request.images)} images and {len(request.labels)} labels.")

        # Добавляем изображения и лейблы в базу данных
        self.face_recognition_ai.add_images(request.images, request.labels)

        print("[INFO] Images and labels processed successfully.")
        return face_recognition_pb2.ImageResponse(success=True, message="Images and labels added to database")

    def GetResults(self, request, context):
        print("[INFO] Received request to get results.")

        # Получаем текущие кадры с камер
        frames = self.face_recognition_ai.get_current_frames()

        # Преобразуем кадры в формат, подходящий для gRPC
        processed_images = [
            cv2.imencode(".jpg", frame)[1].tobytes()
            for all_frames in frames
            for frame in all_frames
            if frame is not None
        ]

        # Расчет FPS
        current_time = time.time()
        if not hasattr(self, "_last_fps_update"):
            # Инициализация при первом вызове
            self._last_fps_update = current_time
            self._frame_counter = 0

        self._frame_counter += len(processed_images)
        time_diff = current_time - self._last_fps_update

        if time_diff >= 1.0:
            fps = self._frame_counter / time_diff
            print(f"[STATS] Frames sent per second: {fps:.2f}")
            self._last_fps_update = current_time
            self._frame_counter = 0

        print(f"[INFO] Returning {len(processed_images)} processed frames.")
        return face_recognition_pb2.ResultResponse(processed_images=processed_images)

    def stop(self):
        """Останавливает поток отображения."""
        self.face_recognition_ai.stop()
        self.camera_thread.join()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FaceRecognitionServicer()
    face_recognition_pb2_grpc.add_FaceRecognitionServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50052')
    print("[INFO] Python gRPC server started on port 50052.")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("[INFO] Shutting down...")
        servicer.face_recognition_ai.stop()
        server.stop(0)

if __name__ == '__main__':
    serve()