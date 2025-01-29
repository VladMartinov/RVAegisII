import time
import cv2
from core.camera import CameraManager

def main():
    camera_manager = CameraManager()
    available_cameras = camera_manager.get_available_cameras()

    if available_cameras:
        print("Доступные камеры:", available_cameras)
        camera_manager.start_capture(available_cameras)
        try:
            prev_time = time.time()
            frame_count = 0
            while True:
                frames = camera_manager.get_frames()
                for camera_index, all_frames in frames.items():
                    for frame in all_frames:
                        cv2.imshow(f"Camera {camera_index}", frame)

                    # Вычисление и вывод FPS
                    current_time = time.time()
                    frame_count += len(all_frames)
                    if (current_time - prev_time) > 1.0:
                        fps = frame_count / (current_time - prev_time)
                        print(f"Camera {camera_index}: FPS = {fps:.2f}")
                        prev_time = current_time
                        frame_count = 0

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nЗавершение работы программы.")
        finally:
            camera_manager.stop_capture()
            cv2.destroyAllWindows()
    else:
        print("Нет доступных камер.")

if __name__ == '__main__':
    main()