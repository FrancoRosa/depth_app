import tkinter as tk
from tkinter import Label, Scale, HORIZONTAL, Frame, Checkbutton, BooleanVar
from PIL import Image, ImageTk
import cv2
import depthai as dai
from yolo_setup import setup_pipeline_yolo, LABEL_MAP_YOLO
import numpy as np
from utility import *
from playsound import playsound
import threading
from queue import Queue

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OAK-D Depth Viewer")
        self.setup_video()
        self.text = TextHelper()
        self.fps = FPSHandler()
        self.NN_SIZE = (640, 640)
        
        self.root.configure(background='#333333')  # Dark background for the app
        
        self.main_frame = Frame(self.root, background='#333333')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=0)  # Slider column
        self.main_frame.grid_columnconfigure(1, weight=1)  # Video frame column
        self.main_frame.grid_columnconfigure(2, weight=0)  # Detection column
        
        self.video_frame = Frame(self.main_frame, background='#222222')
        self.video_frame.grid(row=0, column=1, sticky="nsew")
        
        self.label = Label(self.video_frame, background='#222222')
        self.label.pack(fill=tk.BOTH, expand=True)
        
        self.slider_frame = Frame(self.main_frame, background='#444444')
        self.slider_frame.grid(row=0, column=0, sticky="ns")
        
        self.detection_frame = Frame(self.main_frame, background='#444444')
        self.detection_frame.grid(row=0, column=2, sticky="ns")
        
        self.detection_label = Label(self.detection_frame, text="Detections: 0", font=('Helvetica', 16), fg='white', bg='#444444')
        self.detection_label.pack(pady=20, padx=20)

        # Create checkbox variables
        self.show_humans = BooleanVar(value=True)
        self.show_piles = BooleanVar(value=True)
        self.show_crate = BooleanVar(value=True)
        self.play_sound = BooleanVar(value=True)

        # Create checkboxes
        Checkbutton(self.detection_frame, text="Audio Function", variable=self.play_sound, bg='#444444', fg='white', selectcolor="#333333").pack(anchor='w')
        Checkbutton(self.detection_frame, text="Human Detection", variable=self.show_humans, bg='#444444', fg='white', selectcolor="#333333").pack(anchor='w')
        Checkbutton(self.detection_frame, text="Crate/box", variable=self.show_crate, bg='#444444', fg='white', selectcolor="#333333").pack(anchor='w')
        Checkbutton(self.detection_frame, text="Piles/Tube", variable=self.show_piles, bg='#444444', fg='white', selectcolor="#333333").pack(anchor='w')
        
        self.sliders_initialized = False
        self.setup_sliders()
        self.original_aspect_ratio = None
        self.img = None
        self.imgtk = None
        self.new_height = 800
        self.new_width = 600
        self.resized_imgtk = None
        self.window_resized = False
        self.sound_queue = Queue()
        threading.Thread(target=self.sound_worker, daemon=True).start()
        self.root.after(100, self.update_video)
        self.root.bind("<Configure>", self.on_resize)
        
    def sound_worker(self):
        while True:
            sound_path = self.sound_queue.get()
            playsound(sound_path)
            self.sound_queue.task_done()
        
    def play_sound_async(self, sound_path):
        self.sound_queue.put(sound_path)
        
    def setup_sliders(self):
        self.sliders = {}
        colors = ["Red", "Orange", "Yellow", "Green"]
        for i, color in enumerate(colors):
            slider = Scale(self.slider_frame, from_=0, to=100, orient=HORIZONTAL, label=f'{color} threshold (ft)',
                           bg='#444444', fg='white', troughcolor='#555555', sliderlength=20, highlightbackground='#333333')
            slider.set(6 + i * 9)
            slider.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
            self.sliders[color] = slider
        self.sliders_initialized = True
        
    def on_resize(self, event):
        if self.img and self.original_aspect_ratio:
            self.new_width = event.width
            self.new_height = int(self.new_width / self.original_aspect_ratio)
            if self.new_height > event.height:
                self.new_height = event.height
                self.new_width = int(self.new_height * self.original_aspect_ratio)
            resized_img = self.img.resize((self.new_width, self.new_height), Image.LANCZOS)
            self.resized_imgtk = ImageTk.PhotoImage(image=resized_img)
            self.label.imgtk = self.resized_imgtk
            self.label.config(image=self.resized_imgtk)
            self.window_resized = True

    def setup_video(self):
        self.pipeline = setup_pipeline_yolo()
        self.device = dai.Device(self.pipeline)
        self.queues = {
            'rgb': self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False),
            'detections': self.device.getOutputQueue(name="detections", maxSize=4, blocking=False),
            'depth': self.device.getOutputQueue(name="depth", maxSize=4, blocking=False),
            'pass': self.device.getOutputQueue(name="pass", maxSize=4, blocking=False)
        }

    def frameNorm(self, frame, NN_SIZE, bbox):
        ar_diff = NN_SIZE[0] / NN_SIZE[0] - frame.shape[0] / frame.shape[1]
        sel = 0 if 0 < ar_diff else 1
        bbox[sel::2] *= 1 - abs(ar_diff)
        bbox[sel::2] += abs(ar_diff) / 2
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(bbox, 0, 1) * normVals).astype(int)
        
    def displayFrame(self, name, frame, NN_SIZE, detections):
        critical_alert_triggered = False
        if not self.sliders_initialized:
            return
        
        try:
            depth_thresholds = {
                "Red": self.sliders["Red"].get(),
                "Orange": self.sliders["Orange"].get(),
                "Yellow": self.sliders["Yellow"].get(),
                "Green": self.sliders["Green"].get()
            }
        except KeyError as e:
            print("KeyError in accessing slider:", str(e))
            return  # Skip this frame
            
        self.detection_label.config(text=f"Detections: {len(detections)}")
        for detection in detections:
            label = LABEL_MAP_YOLO[detection.label]
            if (label == "person" and self.show_humans.get()) or (label == "pile" and self.show_piles.get()) or (label == "tube" and self.show_piles.get()) or (label == "crate" and self.show_crate.get()) or (label == "junction box" and self.show_crate.get()):
                bbox = self.frameNorm(frame, NN_SIZE, np.array([detection.xmin, detection.ymin, detection.xmax, detection.ymax]))
                depth_in_mm = int(detection.spatialCoordinates.z)
                depth_in_feet = depth_in_mm / 304.8

                if depth_in_feet < depth_thresholds["Red"]:
                    bbox_color = (255, 0, 0)
                    critical_alert_triggered = True
                    if self.play_sound.get():
                        self.play_sound_async('warn.mp3')
                elif depth_in_feet < depth_thresholds["Orange"]:
                    bbox_color = (255, 165, 0)
                elif depth_in_feet < depth_thresholds["Yellow"]:
                    bbox_color = (255, 255, 0)
                elif depth_in_feet < depth_thresholds["Green"]:
                    bbox_color = (0, 255, 0)
                else:
                    bbox_color = (255, 255, 255)

                self.text.putText(frame, LABEL_MAP_YOLO[detection.label], (bbox[0] + 10, bbox[1] + 20), bbox_color)
                self.text.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), bbox_color)
                self.text.putText(frame, f"Z: {depth_in_mm} mm (~{depth_in_feet:.2f} ft)", (bbox[0] + 10, bbox[1] + 60), bbox_color)
                self.text.rectangle(frame, bbox, bbox_color)
                if critical_alert_triggered:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    
    def update_video(self):
        if not self.sliders_initialized:
            self.root.after(100, self.update_video)
            return
        if all(q.has() for q in self.queues.values()):
            frames = {name: q.get() for name, q in self.queues.items()}

            frame = frames['rgb'].getCvFrame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.original_aspect_ratio is None:
                self.original_aspect_ratio = frame.shape[1] / frame.shape[0]
            
            depthFrame = frames["depth"].getFrame()
            depth_downscaled = depthFrame[::4]
            if np.all(depth_downscaled == 0):
                min_depth = 0
            else:
                min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
            max_depth = np.percentile(depth_downscaled, 99)
            depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
            detections = frames['detections'].detections if frames['detections'] is not None else []

            self.fps.next_iter()
            self.text.putText(frame, "NN fps: {:.2f}".format(self.fps.fps()), (2, frame.shape[0] - 4), (255, 165, 0))
            self.displayFrame("preview", frame, self.NN_SIZE, detections)
            
            self.img = Image.fromarray(frame)
            resized_img = self.img.resize((self.new_width, self.new_height), Image.LANCZOS)
            self.imgtk = ImageTk.PhotoImage(image=resized_img)
            self.label.imgtk = self.imgtk
            self.label.config(image=self.imgtk)

            self.window_resized = False

            if not hasattr(self, 'initial_size_set'):
                self.root.geometry(f"{frame.shape[1]}x{frame.shape[0]}")
                self.initial_size_set = True

        self.root.after(30, self.update_video)

def main():
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
