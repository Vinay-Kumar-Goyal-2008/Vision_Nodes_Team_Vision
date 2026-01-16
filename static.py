import cv2
import numpy as np
import time
import os
import tkinter as tk
from PIL import Image, ImageTk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    HandLandmarker,
    HandLandmarkerOptions
)

# Paths
BASE_DIR = os.path.dirname(__file__)


GESTURE_MODEL = os.path.join(BASE_DIR, "gesture_recognizer.task")
HAND_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")

# config
PINCH_THRESHOLD = 0.04
MIN_CONFIDENCE = 0.75

IGNORED_AS_NONE = {"Open_Palm", "Pointing_Up"}
ALLOWED_GESTURES = {
    "Closed_Fist",
    "Thumb_Up",
    "Thumb_Down",

    "Victory",
    "ILoveYou"
}

# Models
gesture_options = GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path=GESTURE_MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(gesture_options)

hand_options = HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
hand_landmarker = HandLandmarker.create_from_options(hand_options)


# UI APP
class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition • MediaPipe Tasks")
        
        self.root.geometry("950x750")
        self.root.configure(bg="#0f172a")  # Dark blue background

        # ================== HEADER ==================
    
        header = tk.Frame(root, bg="#1e293b", height=60)
        header.pack(fill=tk.X)

        title = tk.Label(
            header,
            text="🤖 Gesture Recognition Dashboard",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 20, "bold")
        )
        title.pack(pady=12)

        # Video frame
        video_frame = tk.Frame(
            
            root,
            bg="#020617",
            bd=3,
            relief=tk.GROOVE
        )
        video_frame.pack(pady=20)

        self.video_label = tk.Label(video_frame, bg="#020617")
    
        self.video_label.pack(padx=10, pady=10)
    
        # ================== INFO PANEL ==================
        info_frame = tk.Frame(root, bg="#0f172a")
        info_frame.pack(pady=10)
        

        self.gesture_label = tk.Label(
            info_frame,
            text="Gesture: NONE",
            font=("Segoe UI", 18, "bold"),
            fg="#22c55e",
            bg="#0f172a"
        )
        self.gesture_label.pack(pady=6)
        
        self.fps_label = tk.Label(
            info_frame,
            text="FPS: 0",
            font=("Segoe UI", 14),
            fg="#facc15",
            bg="#0f172a"
        )
        self.fps_label.pack()

        # ================== BUTTON PANEL ==================
        btns = tk.Frame(root, bg="#0f172a")
        btns.pack(pady=25)

        start_btn = tk.Button(
            
            btns,
            text="▶ START",
            width=14,
            font=("Segoe UI", 12, "bold"),
            bg="#22c55e",
            fg="black",
            activebackground="#16a34a",
            relief=tk.FLAT,
            command=self.start
        )
        start_btn.pack(side=tk.LEFT, padx=15, ipady=6)

        stop_btn = tk.Button(
            btns,
            text="⏹ STOP",
            width=14,
            font=("Segoe UI", 12, "bold"),
            bg="#ef4444",
            fg="white",
            activebackground="#dc2626",
            relief=tk.FLAT,
            command=self.stop
        )
        stop_btn.pack(side=tk.LEFT, padx=15, ipady=6)

        # Status bar
        self.status = tk.Label(
            root,
            text="Status: Idle",
            bg="#020617",
            fg="#94a3b8",
            anchor="w",
            padx=10
        )
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # ================== CAMERA STATE ==================
        self.cap = None
        self.running = False
        self.timestamp = 0
        self.prev_time = time.time()
        # self.prev_time=time.time()#
        # self.pinch

    def start(self):
        if not self.running:
            self.status.config(text="Status: Camera Running")

            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def stop(self):
        self.running = False
        if self.cap:
            self.status.config(text="Status: Stopped")

            self.cap.release()
            self.cap = None

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        # ---------- FPS ----------
        curr = time.time()
        fps = int(1 / max(curr - self.prev_time, 1e-6))
        self.prev_time = curr

        gesture_text = "NONE"
        gesture_score = 0.0
        is_pinch = False

        self.timestamp += 1
    
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Pinch landmarker 
        hand_result = hand_landmarker.detect_for_video(mp_image, self.timestamp)

        if hand_result.hand_landmarks:
            lm = hand_result.hand_landmarks[0]
            thumb = lm[4]
            index = lm[8]

            pinch_dist = np.linalg.norm(
                np.array([thumb.x, thumb.y]) -
                np.array([index.x, index.y])
            )

            if pinch_dist < PINCH_THRESHOLD:

                is_pinch = True
                
            

        #Gesture 
        if is_pinch:

            gesture_text = "PINCH"
        

        else:
            result = gesture_recognizer.recognize_for_video(mp_image, self.timestamp)

            if result.gestures:
                top = result.gestures[0][0]
            
                if (
                    top.category_name in ALLOWED_GESTURES
                    and top.score >= MIN_CONFIDENCE
                ):
                    gesture_text = top.category_name
                    gesture_score = top.score
                    
                    

        # UI update
        if gesture_score:
            self.gesture_label.config(
                text=f"Gesture: {gesture_text} ({gesture_score:.2f})"
            )
        else:
            self.gesture_label.config(text=f"Gesture: {gesture_text}")

        self.fps_label.config(text=f"FPS: {fps}")

        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

        self.root.after(10, self.update_frame)

# ================== RUN ==================
root = tk.Tk()
app = GestureApp(root)
root.mainloop()