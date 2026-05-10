import cv2
import threading
import queue
import time
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

class ThreadedTracker:
    def __init__(self, input_path, output_path, model_name='yolov8n.pt'):
        self.input_path = input_path
        self.output_path = output_path
        self.model = YOLO(model_name)
        
        # Queues for inter-thread communication
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer for raw frames
        self.output_queue = queue.Queue(maxsize=30) # Buffer for annotated frames
        
        self.stop_event = threading.Event()
        
        # Tracking data
        self.track_history = defaultdict(lambda: [])
        self.total_unique_ids = set()

    def video_reader(self):
        """Thread 1: Reads frames from disk and pushes to queue."""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {input_video_path}")
            print("Press Ctrl+C to stop the process")
            
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                self.stop_event.set()
                break
            
            # If queue is full, this will block until space is available
            self.frame_queue.put(frame)
            
        cap.release()
        print("[Reader] Finished reading file.")

    def video_processor(self, fps):
        """Thread 2: The Core Engine. Handles YOLO, Tracking, and Analytics."""
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=2)
            except queue.Empty:
                continue

            # Run Tracking
            results = self.model.track(frame, persist=True, classes=[0], verbose=False, tracker="bytetrack.yaml")
            annotated_frame = results[0].plot()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    self.total_unique_ids.add(track_id)
                    x, y, w, h = box
                    center = (float(x), float(y))
                    
                    # Trajectory tail logic
                    self.track_history[track_id].append(center)
                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id].pop(0)

                    # Draw Tail
                    points = np.hstack(self.track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

                    if len(track_history[track_id]) >= 2:
                    dx = track_history[track_id][-1][0] - track_history[track_id][-2][0]
                    dy = track_history[track_id][-1][1] - track_history[track_id][-2][1]
                    
                    # Pixels per frame * frames per second = Pixels per second
                    speed_px_per_sec = np.sqrt(dx**2 + dy**2) * fps 
                    
                    # Display speed above the bounding box
                    cv2.putText(annotated_frame, f"{speed_px_per_sec:.0f} px/s", 
                                (int(x - w/2), int(y - h/2)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # UI Overlay
            cv2.putText(annotated_frame, f"IDs Tracked: {len(self.total_unique_ids)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.output_queue.put(annotated_frame)
            self.frame_queue.task_done()

    def video_writer(self, fps, width, height):
        """Thread 3: Encodes and writes processed frames to disk."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        while not self.stop_event.is_set() or not self.output_queue.empty():
            try:
                frame = self.output_queue.get(timeout=2)
                out.write(frame)
                self.output_queue.task_done()
            except queue.Empty:
                continue
                
        out.release()
        print("[Writer] Finished writing file.")

    def start(self):
        # Initial capture to get video metadata
        cap = cv2.VideoCapture(self.input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Define and start threads
        t1 = threading.Thread(target=self.video_reader)
        t2 = threading.Thread(target=self.video_processor, args=(fps,))
        t3 = threading.Thread(target=self.video_writer, args=(fps, width, height))

        start_time = time.time()
        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()
        
        print(f"Total Processing Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    tracker = ThreadedTracker("vid.mp4", "threaded_output.mp4")
    tracker.start()
