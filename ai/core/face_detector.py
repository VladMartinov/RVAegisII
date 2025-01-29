import cv2
import numpy as np
import face_recognition
from config import Config

class FaceDetectorBase:
    def detect_faces(self, frame):
        raise NotImplementedError()

class HaarCascadeDetector(FaceDetectorBase):
    def __init__(self, cascade_path, scale_factor = 1.1, min_neighbors = 5):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect_faces(self, frame):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor = self.scale_factor,
            minNeighbors = self.min_neighbors
        )
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]

class SSDDetector(FaceDetectorBase):
    def __init__(self, prototxt_path, model_path, confidence_threshold = 0.7):
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.confidence_threshold = confidence_threshold

    def detect_faces(self, frame):
        (h, w) = frame.shape[:2]

        if len(frame.shape) == 2:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            bgr_frame = frame

        blob = cv2.dnn.blobFromImage(cv2.resize(bgr_frame, (300, 300)), 1.0, 
                                   (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
        return faces

class FaceRecognitionDetector(FaceDetectorBase):
    def __init__(self, model = "hog", number_of_times_to_upsample = 1):
        self.model = model
        self.number_of_times_to_upsample = number_of_times_to_upsample

    def detect_faces(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(
            rgb_image,
            model = self.model,
            number_of_times_to_upsample = self.number_of_times_to_upsample
        )
        return [(left, top, right, bottom) for (top, right, bottom, left) in face_locations]

class FaceDetector:
    def __init__(self):
        cfg = Config().FACE_DETECTOR
        self.detector_type = cfg["type"]
        
        if self.detector_type == "haarcascade":
            params = cfg["haarcascade"]
            self.detector = HaarCascadeDetector(
                cascade_path = params["cascade_path"],
                scale_factor = params["scale_factor"],
                min_neighbors = params["min_neighbors"]
            )
        elif self.detector_type == "ssd":
            params = cfg["ssd"]
            self.detector = SSDDetector(
                prototxt_path = params["prototxt_path"],
                model_path = params["model_path"],
                confidence_threshold = params["confidence_threshold"]
            )
        elif self.detector_type == "face_recognition":
            params = cfg["face_recognition"]
            self.detector = FaceRecognitionDetector(
                model = params["model"],
                number_of_times_to_upsample = params["number_of_times_to_upsample"]
            )
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

    def detect_faces(self, frame):
        return self.detector.detect_faces(frame)