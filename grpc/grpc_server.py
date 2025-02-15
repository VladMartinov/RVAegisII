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

        # Преобразуем кадры в формат для gRPC с группировкой по камерам
        processed_camera_frames = []
        total_frames = 0

        for camera_index, all_frames in frames.items():
            encoded_frames = []
            for frame in all_frames:
                if frame is not None:
                    # Кодируем кадр в JPEG
                    success, buffer = cv2.imencode(".jpg", frame)
                    if success:
                        encoded_frames.append(buffer.tobytes())

            # Добавляем данные камеры в результат
            processed_camera_frames.append(
                face_recognition_pb2.CameraFrames(
                    camera_index=camera_index,
                    frames=encoded_frames
                )
            )

            total_frames += len(encoded_frames)

        # Расчет FPS
        current_time = time.time()
        if not hasattr(self, "_last_fps_update"):
            self._last_fps_update = current_time
            self._frame_counter = 0

        self._frame_counter += total_frames
        time_diff = current_time - self._last_fps_update

        if time_diff >= 1.0:
            fps = self._frame_counter / time_diff
            print(f"[STATS] Frames sent per second: {fps:.2f}")
            self._last_fps_update = current_time
            self._frame_counter = 0

        print(f"[INFO] Returning {total_frames} frames from {len(processed_camera_frames)} cameras.")
        return face_recognition_pb2.ResultResponse(
            camera_frames=processed_camera_frames,
            recognized_labels=[]  # Заглушка, можно реализовать логику
        )

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