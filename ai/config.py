import os
import cv2

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Настройки камер
    CAMERA_RESOLUTION = (640, 480)
    CAMERA_FPS = 30
    MAX_FRAMES_IN_QUEUE = 5
    SHOW_CAMERA_WINDOW = True
    NUM_CAMERAS = 2

    # Настройки обработки
    IMAGE_PROCESSORS = ['grayscale', 'blur', 'face_detect']

    # Настройки детекции лиц
    FACE_DETECTOR = {
        "type": "ssd",
        "haarcascade": {
            "cascade_path": cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            "scale_factor": 1.1,
            "min_neighbors": 5
        },
        "ssd": {
            "prototxt_path": os.path.join(BASE_DIR, "ai", "core", "models", "deploy.prototxt"),
            "model_path": os.path.join(BASE_DIR, "ai", "core", "models", "res10_300x300_ssd_iter_140000.caffemodel"),
            "confidence_threshold": 0.7
        },
        "face_recognition": {
            "model": "hog",
            "number_of_times_to_upsample": 1
        }
    }

    # Настройки сравнения лиц
    FACE_COMPARISON = {
        "method": "face_recognition",
        "face_recognition": {
            "model": "small",
            "num_jitters": 1,
            "tolerance": 0.6
        },
        "deepface": {
            "model": "Facenet512",
            "metric": "cosine",
            "detector_backend": "opencv"
        }
    }

# Экземпляр конфигурации
config = Config()