import threading

class FaceDatabase:
    def __init__(self):
        self.known_faces = {}
        self.lock = threading.RLock()

    def add_face(self, face_id, face_encoding):
        with self.lock:
            self.known_faces[face_id] = face_encoding

    def remove_face(self, face_id):
        with self.lock:
            if face_id in self.known_faces:
                del self.known_faces[face_id]

    def get_face_encodings(self):
        with self.lock:
            return list(self.known_faces.values())

    def get_face_ids(self):
        with self.lock:
            return list(self.known_faces.keys())

    def clear(self):
        with self.lock:
            self.known_faces.clear()