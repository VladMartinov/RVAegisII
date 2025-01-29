import numpy as np
from config import Config
from deepface import DeepFace
import face_recognition

class FaceComparer:
    def get_face_encodings(self, image):
        raise NotImplementedError()
    
    def compare_faces(self, face_encodings, known_face_encodings):
        raise NotImplementedError()

class FaceRecognitionComparer(FaceComparer):
    def __init__(self):
        cfg = Config().FACE_COMPARISON["face_recognition"]
        self.model = cfg["model"]
        self.num_jitters = cfg["num_jitters"]
        self.tolerance = cfg["tolerance"]

    def get_face_encodings(self, image):
        rgb_image = np.array(image[:, :, ::-1])  # BGR to RGB
        return face_recognition.face_encodings(
            rgb_image,
            num_jitters = self.num_jitters,
            model = self.model
        )

    def compare_faces(self, face_encodings, known_face_encodings):
        if not face_encodings or not known_face_encodings:
            return []
            
        return face_recognition.compare_faces(
            known_face_encodings,
            face_encodings[0],
            tolerance = self.tolerance
        )

class DeepFaceComparer(FaceComparer):
    def __init__(self):
        cfg = Config().FACE_COMPARISON["deepface"]
        self.model_name = cfg["model"]
        self.metric = cfg["metric"]
        self.detector_backend = cfg["detector_backend"]

    def get_face_encodings(self, image):
        try:
            result = DeepFace.represent(
                img_path = image,
                model_name = self.model_name,
                detector_backend = self.detector_backend,
                enforce_detection = False
            )
            return [np.array(r["embedding"]) for r in result] if result else []
        except Exception as e:
            print(f"DeepFace error: {str(e)}")
            return []

    def compare_faces(self, face_encodings, known_face_encodings):
        if not face_encodings or not known_face_encodings:
            return []

        distances = []
        for known_enc in known_face_encodings:
            result = DeepFace.verify(
                img1_path = None,
                img2_path = None,
                model_name = self.model_name,
                distance_metric = self.metric,
                detector_backend = self.detector_backend,
                embeddings = [face_encodings[0], known_enc]
            )
            distances.append(result["distance"])

        # Нормализация расстояний в булевы значения
        threshold = self._get_threshold()
        return [d <= threshold for d in distances]

    def _get_threshold(self):
        # Пороговые значения для разных моделей
        thresholds = {
            "VGG-Face": 0.55,
            "Facenet": 0.35,
            "Facenet512": 0.15,
            "OpenFace": 0.2,
            "ArcFace": 0.3
        }
        return thresholds.get(self.model_name, 0.4)

class FaceRecognizer:
    def __init__(self):
        method = Config().FACE_COMPARISON["method"]

        if method == "face_recognition":
            self.comparer = FaceRecognitionComparer()
        elif method == "deepface":
            self.comparer = DeepFaceComparer()
        else:
            raise ValueError(f"Unknown comparison method: {method}")

    def get_face_encodings(self, image):
        return self.comparer.get_face_encodings(image)

    def compare_faces(self, face_encodings, known_face_encodings):
        return self.comparer.compare_faces(face_encodings, known_face_encodings)