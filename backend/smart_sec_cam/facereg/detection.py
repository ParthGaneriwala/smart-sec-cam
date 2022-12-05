import queue
import threading
import time
from typing import List

import cv2
import imutils
import numpy as np
from deepface import DeepFace as df
from smart_sec_cam.video.writer import VideoWriter


class FaceDetector:
    def __init__(self, channel_name: str, face_threshold: int = 0.40, video_duration_seconds: int = 10,
                 video_dir: str = "data/videos"):
        self.channel_name = channel_name
        self.face_threshold = face_threshold
        self.video_duration = video_duration_seconds
        self.video_dir = video_dir
        self.video_writer = VideoWriter(self.channel_name, path=self.video_dir)
        self.frame_queue = queue.Queue()
        self.detection_thread = threading.Thread(target=self.run, daemon=True)
        self.shutdown = False

    def add_frame(self, frame: bytes):
        self.frame_queue.put(frame)

    def run(self):
        last_frame = None
        last_frame_greyscale = None
        recorded_video = False
        while not self.shutdown:
            decoded_frame = self._get_decoded_frame()
            cv2.imwrite("frame.png", decoded_frame)
            decoded_frame_path="frame.png"

            if last_frame is not None:
                if self._detect_face(decoded_frame_path):
                    print("Detected face for channel: " + self.channel_name)
                    self._record_video([last_frame, decoded_frame])
                    recorded_video = True
            # Set current frame to last frame
            if not recorded_video:
                last_frame = decoded_frame
                last_frame_greyscale = decoded_frame
            else:
                print("Done recording video for channel: " + str(self.channel_name))
                last_frame = None
                last_frame_greyscale = None
                recorded_video = False

    def run_in_background(self):
        self.detection_thread.start()

    def stop(self):
        self.shutdown = True

    def _has_decoded_frame(self) -> bool:
        return not self.frame_queue.empty()

    def _get_decoded_frame(self, greyscale=False):
        new_frame = self.frame_queue.get()
        if greyscale:
            return self._decode_frame_greyscale(new_frame)
        else:
            return self._decode_frame(new_frame)

    def _detect_face(self,new_frame_path) -> bool:
        """
        Face Detection and Verification.
        Returns a boolean indicating if the face detected and verified.
        """
        metrics = ["cosine", "euclidean", "euclidean_l2"]

        #face recognition
        result = df.find(new_frame_path, 
                db_path = "", 
                distance_metric = metrics[2])

        if result["verified"] == True:
            return True
        return False

    def _draw_face_areas_on_frame(self, old_frame, new_frame):
        if old_frame is None:
            return new_frame
        # Convert frames to greyscale
        old_frame_greyscale = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        new_frame_greyscale = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # Calculate background subtraction
        frame_delta = cv2.absdiff(old_frame_greyscale, new_frame_greyscale)
        # Calculate and dilate threshold
        threshold = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None, iterations=2)
        # Extract contours from the threshold image
        contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        # Iterate over contours and determine if any are large enough to count as face
        modified_frame = new_frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) >= self.face_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(modified_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return modified_frame

    def _record_video(self, first_frames: List):
        start_time = time.monotonic()
        self.video_writer.reset()
        old_frame = None
        for frame in first_frames:
            self.video_writer.add_frame(frame)
            old_frame = frame
        while not self._done_recording_video(start_time):
            if self._has_decoded_frame():
                new_frame = self._get_decoded_frame()
                new_frame_with_face_area = self._draw_face_areas_on_frame(old_frame, new_frame)
                self.video_writer.add_frame(new_frame_with_face_area)
                old_frame = new_frame
            else:
                time.sleep(0.01)
        self.video_writer.write()

    def _done_recording_video(self, start_time: float) -> bool:
        return time.monotonic() - start_time > self.video_duration

    @staticmethod
    def _decode_frame(frame: bytes):
        return cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def _decode_frame_greyscale(frame: bytes):
        # Convert frame to greyscale and blur it
        greyscale_frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        return cv2.GaussianBlur(greyscale_frame, (21, 21), 0)
