import cv2
import time

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords,color):
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.5, color, 1, self.line_type)
    def rectangle(self, frame, bbox, color):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, 3)  # thicker border in background color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # colored border
        # print(f"Assigned color: {color}")
class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)