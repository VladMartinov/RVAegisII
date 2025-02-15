import queue

class FrameQueue:
    def __init__(self, max_size):
        self.queue = queue.Queue(maxsize=max_size)

    def put(self, frame):
        try:
            self.queue.put(frame, block=False)
        except queue.Full:
            pass

    def get_all(self):
        frames = []
        try:
            while True:
                frame = self.queue.get(block = False)
                frames.append(frame)
        except queue.Empty:
            pass
        return frames

    def clear(self):
        with self.queue.mutex:
            self.queue.queue.clear()

    def is_empty(self):
        return self.queue.empty()
