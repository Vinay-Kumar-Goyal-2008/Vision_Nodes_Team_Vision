
import cv2
import numpy as np
import time
import os
import threading
import tkinter as tk
from PIL import Image, ImageTk
from queue import Queue, Empty
from collections import deque, Counter
from typing import Optional, Dict, Tuple, List
import pyttsx3

import torch
import torch.nn as nn

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    HandLandmarker,
    HandLandmarkerOptions
)

import speech_recognition as sr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from playwright.sync_api import sync_playwright
import requests
import pyautogui  # Single import at top

# Initialize pyttsx3 engine once (more efficient)
TTS_ENGINE = pyttsx3.init()

# API Base URL for OnDemand agents
API_BASE_URL = "http://127.0.0.1:5000"


BASE_DIR = os.path.dirname(__file__)

# --- Model Paths ---
GESTURE_MODEL = os.path.join(BASE_DIR, "gesture_recognizer.task")
HAND_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")
DYNAMIC_GESTURE_MODEL_PATH = os.path.join(BASE_DIR, "dynamic_gesture_model.pth")
SENTENCE_MODEL_PATH = os.path.join(BASE_DIR, "all-MiniLM-L6-v2")

# --- Normalization Thresholds ---
T_GESTURE = 0.8
T_SPEECH = 0.45

# --- Gesture Detection Config ---
PINCH_THRESHOLD = 0.04
MIN_STATIC_CONFIDENCE = 0.75
BUFFER_SIZE = 30
NUM_FRAMES = 20
D_MODEL = 32
VELOCITY_THRESHOLD = 0.01
COOLDOWN = 1.5
DYNAMIC_CONFIDENCE_THRESHOLD = 0.85
VOTE_CONSENSUS = 12
CONFIRMED_DISPLAY_TIME = 2.0
STATIC_COOLDOWN = 3.0  # 3 second cooldown for static gestures

# --- Playwright Config ---
AUTH_FILE = os.path.join(BASE_DIR, "auth.json")
HEADLESS_MODE = False
SLOW_MO = 50  # Reduced from 500ms to fix lag

# --- Gesture Classes ---
ALLOWED_STATIC_GESTURES = {
    "Closed_Fist", "Thumb_Up", "Thumb_Down", "Victory", "ILoveYou", "Open_Palm"
}

DYNAMIC_GESTURE_CLASSES = {
    0: "none", 1: "swipe_down", 2: "swipe_left", 3: "swipe_right", 4: "swipe_up"
}

# --- Voice Commands ---
VOICE_COMMANDS = {
    "navigation": ["open youtube", "open gmail", "open wikipedia", "go back", "refresh page"],
    "youtube": ["play video", "pause video", "skip ad", "volume up", "volume down", "mute", "forward", "rewind"],
    "gmail": ["compose email", "refresh inbox", "go to spam", "archive email", "go to inbox", "filter unread"],
    "system": ["voice typing", "read aloud"],
    'ondemand':["create web summary","send the email","health chat","wikipedia","google books","air quality","media upload","chatbot"]
}

def normalize_gesture_score(raw_score: float) -> float:
    if raw_score < T_GESTURE:
        return -1.0
    return (raw_score - T_GESTURE) / (1 - T_GESTURE)

def normalize_speech_score(raw_score: float) -> float:
    if raw_score < T_SPEECH:
        return -1.0
    return (raw_score - T_SPEECH) / (1 - T_SPEECH)

class NanoTransformer(nn.Module):
    def __init__(self, num_frames=20, num_classes=5, input_dim=63, d_model=32, num_heads=4):
        super().__init__()
        self.feature_extractor = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=num_heads, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

def gated_max_selection(gesture_result, voice_result):
    g_action, g_score = gesture_result if gesture_result else (None, -1.0)
    v_action, v_score = voice_result if voice_result else (None, -1.0)
    
    if g_score > v_score and g_score > 0:
        return ("gesture", g_action, g_score)
    elif v_score > g_score and v_score > 0:
        return ("voice", v_action, v_score)
    elif g_score == v_score and g_score > 0:
        return ("gesture", g_action, g_score)
    return (None, None, -1.0)

def get_current_app(page) -> str:
    if page is None:
        return "unknown"
    try:
        url = page.url.lower()
    except:
        return "unknown"
    
    if "youtube.com" in url:
        return "youtube"
    elif "mail.google.com" in url:
        return "gmail"
    elif "wikipedia.org" in url:
        return "wikipedia"
    return "unknown"

GESTURE_NAME_MAP = {
    "Closed_Fist": "fist", "Thumb_Up": "thumbs_up", "Thumb_Down": "thumbs_down",
    "Victory": "peace", "ILoveYou": "love", "Open_Palm": "open_palm", "PINCH": "pinch",
    "swipe_up": "swipe_up", "swipe_down": "swipe_down", "swipe_left": "swipe_left", "swipe_right": "swipe_right",
}

def intentmap(page, gesture: Optional[str] = None) -> str:
    app = get_current_app(page)
    normalized_gesture = GESTURE_NAME_MAP.get(gesture, gesture)
    if normalized_gesture:
        normalized_gesture = normalized_gesture.lower()
    
    commands = {
        "youtube": {
            "fist": "TOGGLE_PLAY_PAUSE", "thumbs_up": "VOLUME_UP_25", "thumbs_down": "VOLUME_DOWN_25",
            "pinch": "PLAY_FIRST_VIDEO", "open_palm": "VOICE_SEARCH", "peace": "SKIP_AD", "love": "MUTE_TOGGLE",
            "swipe_up": "VOLUME_UP_25", "swipe_down": "VOLUME_DOWN_25", "swipe_left": "REWIND_10S", "swipe_right": "FORWARD_10S",
        },
        "gmail": {
            "fist": "COMPOSE_EMAIL", "peace": "OPEN_SPAM", "open_palm": "REFRESH_INBOX",
            "thumbs_up": "ARCHIVE_EMAIL", "pinch": "JUMP_TO_INBOX", "thumbs_down": "FILTER_UNREAD",
            "swipe_left": "PREVIOUS_EMAIL", "swipe_right": "NEXT_EMAIL", "swipe_up": "SCROLL_UP", "swipe_down": "SCROLL_DOWN",
        },
        "wikipedia": {
            "swipe_up": "SCROLL_UP", "swipe_down": "SCROLL_DOWN", "fist": "GO_BACK",
            "swipe_left": "GO_BACK", "swipe_right": "GO_FORWARD",
        },
        "unknown": {"swipe_up": "SCROLL_UP", "swipe_down": "SCROLL_DOWN", "fist": "GO_BACK"}
    }
    
    if normalized_gesture and normalized_gesture in commands.get(app, {}):
        return commands[app][normalized_gesture]
    if normalized_gesture and normalized_gesture in commands.get("unknown", {}):
        return commands["unknown"][normalized_gesture]
    return "NO_ACTION"


def focus_player(page):
    try:
        page.locator("video").first.click()
        time.sleep(0.1)
    except:
        pass

def execute_action(page, action: str) -> bool:
    if page is None or action == "NO_ACTION":
        return False
    
    try:
        # ===== PLAYBACK CONTROLS =====
        if action in ["TOGGLE_PLAY_PAUSE", "play video", "pause video"]:
            focus_player(page)
            page.keyboard.press('k')
        
        # ===== VOLUME CONTROLS (consolidated) =====
        elif action in ["VOLUME_UP_25", "volume up"]:
            for _ in range(5):
                pyautogui.press('volumeup')
                time.sleep(0.05)
        elif action in ["VOLUME_DOWN_25", "volume down"]:
            for _ in range(5):
                pyautogui.press('volumedown')
                time.sleep(0.05)
        elif action in ["MUTE_TOGGLE", "mute"]:
            pyautogui.press('volumemute')
        
        # ===== YOUTUBE SPECIFIC =====
        elif action == "PLAY_FIRST_VIDEO":
            page.locator("ytd-video-renderer").first.click()
        elif action in ["SKIP_AD", "skip ad"]:
            try:
                btn = page.locator(".ytp-ad-skip-button, .ytp-ad-overlay-close-button")
                btn.wait_for(state="visible", timeout=3000)
                btn.click()
            except:
                pass
        elif action == "VOICE_SEARCH":
            # Works on both YouTube and Google
            url = page.url.lower()
            if "youtube.com" in url:
                page.locator("button[aria-label='Search with your voice']").click()
            elif "google.com" in url:
                # Click microphone icon on Google search
                page.locator("div.XDyW0e, button[aria-label='Search by voice']").first.click()
            else:
                pyautogui.hotkey('win', 'h')  # Windows voice typing fallback
        elif action == "FORWARD_10S":
            focus_player(page)
            page.keyboard.press('l')
        elif action == "REWIND_10S":
            focus_player(page)
            page.keyboard.press('j')
        
        # ===== EMAIL CONTROLS (consolidated) =====
        elif action in ["COMPOSE_EMAIL", "compose email"]:
            page.keyboard.press('c')
        elif action == "ARCHIVE_EMAIL":
            page.keyboard.press('e')
        elif action in ["JUMP_TO_INBOX", "go to inbox"]:
            page.keyboard.press('g')
            time.sleep(0.1)
            page.keyboard.press('i')
        elif action in ["OPEN_SPAM", "go to spam"]:
            page.goto("https://mail.google.com/mail/u/0/#spam")
        elif action == "REFRESH_INBOX":
            page.goto("https://mail.google.com/mail/u/0/#inbox")
        elif action == "FILTER_UNREAD":
            page.goto("https://mail.google.com/mail/u/0/#search/label%3Aunread")
        elif action == "PREVIOUS_EMAIL":
            page.keyboard.press('k')
        elif action == "NEXT_EMAIL":
            page.keyboard.press('j')
        
        # ===== NAVIGATION =====
        elif action == "SCROLL_UP":
            page.keyboard.press('PageUp')
        elif action == "SCROLL_DOWN":
            page.keyboard.press('PageDown')
        elif action == "GO_BACK":
            page.go_back()
        elif action == "GO_FORWARD":
            page.go_forward()
        elif action == "open youtube":
            page.goto("https://www.youtube.com", wait_until="networkidle")
            time.sleep(1)  # Wait for page to fully load
            # Auto-trigger voice search
            try:
                voice_btn = page.locator("button[aria-label='Search with your voice']")
                voice_btn.wait_for(state="visible", timeout=5000)
                voice_btn.click()
                print("[YouTube] Voice search triggered")
                time.sleep(3)  # Wait for voice input to complete
                # After voice search, play first video
                first_video = page.locator("ytd-video-renderer, ytd-rich-item-renderer").first
                first_video.wait_for(state="visible", timeout=10000)
                first_video.click()
                print("[YouTube] First video clicked")
            except Exception as e:
                print(f"[YouTube] Auto-action failed: {e}")
        elif action == "open gmail":
            page.goto("https://mail.google.com")
        elif action == "open wikipedia":
            page.goto("https://en.wikipedia.org")
        
        # ===== SYSTEM =====
        elif action == "voice typing":
            pyautogui.hotkey('win', 'h')
        
        # ===== ONDEMAND AGENTS (fixed API URLs, use TTS_ENGINE) =====
        elif action == "create web summary":
            payload = {"links": [page.url.lower()]}
            try:
                response = requests.post(f"{API_BASE_URL}/web_summary", json=payload, timeout=60)
                result = response.json().get("result", "No summary available")
                print(f"[Agent] Web Summary: {result[:200]}...")
                TTS_ENGINE.say(result)
                TTS_ENGINE.runAndWait()
            except Exception as e:
                print(f"[Agent] Error: {e}")
                TTS_ENGINE.say("Sorry, could not get web summary")
                TTS_ENGINE.runAndWait()
        elif action == "send the email":
            try:
                response = requests.post(f"{API_BASE_URL}/send_email", json={}, timeout=60)
                result = response.json().get("result", "Email sent")
                print(f"[Agent] Email: {result}")
                TTS_ENGINE.say(result)
                TTS_ENGINE.runAndWait()
            except Exception as e:
                print(f"[Agent] Error: {e}")
                TTS_ENGINE.say("Sorry, could not send email")
                TTS_ENGINE.runAndWait()
        elif action == "health chat":
            try:
                response = requests.post(f"{API_BASE_URL}/health_chat", json={}, timeout=60)
                result = response.json().get("result", "No response")
                print(f"[Agent] Health Chat: {result[:200]}...")
                TTS_ENGINE.say(result)
                TTS_ENGINE.runAndWait()
            except Exception as e:
                print(f"[Agent] Error: {e}")
                TTS_ENGINE.say("Sorry, health chat failed")
                TTS_ENGINE.runAndWait()
        elif action == "wikipedia":
            payload = {"links": ["earth"]}
            try:
                response = requests.post(f"{API_BASE_URL}/wikipedia", json=payload, timeout=60)
                result = response.json().get("result", "No response")
                print(f"[Agent] Wikipedia: {result[:200]}...")
                TTS_ENGINE.say(result)
                TTS_ENGINE.runAndWait()
            except Exception as e:
                print(f"[Agent] Error: {e}")
                TTS_ENGINE.say("Sorry, wikipedia search failed")
                TTS_ENGINE.runAndWait()
        elif action == "google books":
            payload = {"links": ["earth"]}
            try:
                response = requests.post(f"{API_BASE_URL}/google_books", json=payload, timeout=60)
                result = response.json().get("result", "No response")
                print(f"[Agent] Google Books: {result[:200]}...")
                TTS_ENGINE.say(result)
                TTS_ENGINE.runAndWait()
            except Exception as e:
                print(f"[Agent] Error: {e}")
                TTS_ENGINE.say("Sorry, google books search failed")
                TTS_ENGINE.runAndWait()
        elif action == "air quality":
            payload = {"links": ["Delhi"]}
            try:
                response = requests.post(f"{API_BASE_URL}/air_quality", json=payload, timeout=60)
                result = response.json().get("result", "No response")
                print(f"[Agent] Air Quality: {result[:200]}...")
                TTS_ENGINE.say(result)
                TTS_ENGINE.runAndWait()
            except Exception as e:
                print(f"[Agent] Error: {e}")
                TTS_ENGINE.say("Sorry, air quality check failed")
                TTS_ENGINE.runAndWait()
        elif action == "media upload":
            try:
                response = requests.post(f"{API_BASE_URL}/media_upload", json={}, timeout=60)
                result = response.json().get("result", "Upload complete")
                print(f"[Agent] Media Upload: {result}")
                TTS_ENGINE.say(result)
                TTS_ENGINE.runAndWait()
            except Exception as e:
                print(f"[Agent] Error: {e}")
                TTS_ENGINE.say("Sorry, media upload failed")
                TTS_ENGINE.runAndWait()
        elif action == "chatbot":
            payload = {"links": ["what is brainwave"]}
            try:
                response = requests.post(f"{API_BASE_URL}/chatbot", json=payload, timeout=60)
                result = response.json().get("result", "No response")
                print(f"[Agent] Chatbot: {result[:200]}...")
                TTS_ENGINE.say(result)
                TTS_ENGINE.runAndWait()
            except Exception as e:
                print(f"[Agent] Error: {e}")
                TTS_ENGINE.say("Sorry, chatbot failed")
                TTS_ENGINE.runAndWait()
        else:
            return False
        
        print(f"[Executor] ✓ {action}")
        return True
    except Exception as e:
        print(f"[Executor] Error: {e}")
        return False


class VoiceThread(threading.Thread):
    def __init__(self, result_queue: Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.model = None
        self.cmd_embeddings = None
        self.all_commands = []
        self.recognizer = None
        self.mic = None
    
    def _initialize(self):
        """Heavy initialization - runs in background thread."""
        print("[VoiceThread] Starting initialization...")
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.8
        self.mic = sr.Microphone()
        
        print("[VoiceThread] Calibrating microphone...")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("[VoiceThread] Loading MiniLM...")
        self.model = SentenceTransformer(SENTENCE_MODEL_PATH)
        
        for cat, cmd_list in VOICE_COMMANDS.items():
            self.all_commands.extend(cmd_list)
        self.cmd_embeddings = self.model.encode(self.all_commands)
        print("[VoiceThread] Ready")
    
    def _get_best_command(self, text: str):
        text_emb = self.model.encode([text])
        scores = cosine_similarity(text_emb, self.cmd_embeddings)[0]
        best_idx = np.argmax(scores)
        return self.all_commands[best_idx], float(scores[best_idx])
    
    def run(self):
        # Initialize in background thread (non-blocking!)
        try:
            self._initialize()
        except Exception as e:
            print(f"[VoiceThread] Init error: {e}")
            return
        
        print("[VoiceThread] Listening...")
        while not self.stop_event.is_set():
            try:
                with self.mic as source:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio)
                print(f"[Voice] Heard: '{text}'")
                best_cmd, raw_score = self._get_best_command(text)
                norm_score = normalize_speech_score(raw_score)
                if norm_score > 0:
                    self.result_queue.put(("voice", best_cmd, norm_score))
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"[Voice] API error: {e}")
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[Voice] Error: {e}")
        print("[VoiceThread] Stopped")

class MultimodalApp:
    """
    Tkinter application for multimodal control with video display.
    Combines gesture recognition, voice commands, and web automation.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Multimodal Controller • Gesture + Voice + Web")
        self.root.geometry("1100x900")
        self.root.configure(bg="#0f172a")
        
        # ================== HEADER ==================
        header = tk.Frame(root, bg="#1e293b", height=60)
        header.pack(fill=tk.X)
        
        title = tk.Label(header, text="🤖 Multimodal Fusion Controller",
                        bg="#1e293b", fg="#38bdf8", font=("Segoe UI", 20, "bold"))
        title.pack(pady=12)
        
        # ================== VIDEO FRAME ==================
        video_frame = tk.Frame(root, bg="#020617", bd=3, relief=tk.GROOVE)
        video_frame.pack(pady=10)
        
        self.video_label = tk.Label(video_frame, bg="#020617")
        self.video_label.pack(padx=10, pady=10)
        
        # ================== INFO PANEL ==================
        info_frame = tk.Frame(root, bg="#0f172a")
        info_frame.pack(pady=5)
        
        self.mode_label = tk.Label(info_frame, text="MODE: IDLE",
                                   font=("Segoe UI", 16, "bold"), fg="#a855f7", bg="#0f172a")
        self.mode_label.pack(pady=4)
        
        self.static_label = tk.Label(info_frame, text="STATIC: NONE",
                                     font=("Segoe UI", 14), fg="#22c55e", bg="#0f172a")
        self.static_label.pack(pady=2)
        
        self.dynamic_label = tk.Label(info_frame, text="DYNAMIC: NONE",
                                      font=("Segoe UI", 14), fg="#f97316", bg="#0f172a")
        self.dynamic_label.pack(pady=2)
        
        self.voice_label = tk.Label(info_frame, text="VOICE: NONE",
                                    font=("Segoe UI", 14), fg="#3b82f6", bg="#0f172a")
        self.voice_label.pack(pady=2)
        
        self.velocity_label = tk.Label(info_frame, text="VELOCITY: 0.000 | GATE: CLOSED",
                                       font=("Segoe UI", 12), fg="#94a3b8", bg="#0f172a")
        self.velocity_label.pack(pady=2)
        
        self.action_label = tk.Label(info_frame, text="ACTION: ---",
                                     font=("Segoe UI", 14, "bold"), fg="#facc15", bg="#0f172a")
        self.action_label.pack(pady=4)
        
        self.fps_label = tk.Label(info_frame, text="FPS: 0",
                                  font=("Segoe UI", 14), fg="#facc15", bg="#0f172a")
        self.fps_label.pack(pady=4)
        
        # ================== BUTTON PANEL ==================
        btns = tk.Frame(root, bg="#0f172a")
        btns.pack(pady=15)
        
        start_btn = tk.Button(btns, text="▶ START", width=14, font=("Segoe UI", 12, "bold"),
                             bg="#22c55e", fg="black", activebackground="#16a34a",
                             relief=tk.FLAT, command=self.start)
        start_btn.pack(side=tk.LEFT, padx=15, ipady=6)
        
        stop_btn = tk.Button(btns, text="⏹ STOP", width=14, font=("Segoe UI", 12, "bold"),
                            bg="#ef4444", fg="white", activebackground="#dc2626",
                            relief=tk.FLAT, command=self.stop)
        stop_btn.pack(side=tk.LEFT, padx=15, ipady=6)
        
        # ================== STATUS BAR ==================
        self.status = tk.Label(root, text="Status: Idle • Press START to begin",
                              bg="#020617", fg="#94a3b8", anchor="w", padx=10)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        
        # ================== STATE VARIABLES ==================
        self.cap = None
        self.running = False
        self.timestamp = 0
        self.prev_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Gesture state
        self.landmark_buffer = deque(maxlen=BUFFER_SIZE)
        self.vote_buffer = deque(maxlen=15)
        self.prev_landmarks = None
        self.last_detection_time = 0
        self.confirmed_dynamic_gesture = None
        self.confirmed_dynamic_time = 0
        
        # Thread communication
        self.result_queue = Queue()
        self.stop_event = threading.Event()
        self.voice_thread = None
        
        # Playwright (will be initialized on start)
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # Models (will be initialized on start)
        self.gesture_recognizer = None
        self.hand_landmarker = None
        self.dynamic_model = None
        
        # Last action display
        self.last_action = None
        self.last_action_time = 0
        
        # Recent results for fusion
        self.gesture_result = None
        self.voice_result = None
        self.last_gesture_time = 0
        self.last_voice_time = 0
        
        # Action queue for non-blocking Playwright execution
        self.action_queue = Queue()
        self.action_busy = False
        
        # Static gesture cooldown tracking
        self.last_static_gesture_time = 0
    
    def _load_models(self):
        """Initialize all models."""
        self.status.config(text="Status: Loading gesture recognizer...")
        self.root.update()
        
        # MediaPipe Gesture Recognizer
        gesture_options = GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=GESTURE_MODEL),
            running_mode=vision.RunningMode.VIDEO, num_hands=1
        )
        self.gesture_recognizer = GestureRecognizer.create_from_options(gesture_options)
        self.root.update()  # Keep responsive
        
        self.status.config(text="Status: Loading hand landmarker...")
        self.root.update()
        
        # MediaPipe Hand Landmarker
        hand_options = HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO, num_hands=1
        )
        self.hand_landmarker = HandLandmarker.create_from_options(hand_options)
        self.root.update()  # Keep responsive
        
        self.status.config(text="Status: Loading dynamic model...")
        self.root.update()
        
        # Dynamic gesture model
        self.dynamic_model = NanoTransformer(num_frames=NUM_FRAMES).to(self.device)
        if os.path.exists(DYNAMIC_GESTURE_MODEL_PATH):
            self.dynamic_model.load_state_dict(
                torch.load(DYNAMIC_GESTURE_MODEL_PATH, map_location=self.device, weights_only=True)
            )
        self.dynamic_model.eval()
        
        self.status.config(text="Status: Models loaded")
        self.root.update()
    
    def start(self):
        """Start the multimodal controller."""
        if not self.running:
            self.status.config(text="Status: Starting... Please wait")
            self.root.update()
            
            # Start camera first (fast)
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.running = True
            
            # Load models with UI updates
            try:
                self._load_models()
            except Exception as e:
                self.status.config(text=f"Status: Model error - {e}")
                return
            
            # Start voice thread (initializes in background)
            self.stop_event.clear()
            self.voice_thread = VoiceThread(self.result_queue, self.stop_event)
            self.voice_thread.start()
            
            # Initialize Playwright in background thread
            self.browser_ready = False
            browser_thread = threading.Thread(target=self._init_playwright_async, daemon=True)
            browser_thread.start()
            
            self.status.config(text="Status: RUNNING • Browser loading...")
            self.root.update()
            
            # Start frame loop
            self.update_frame()
    
    def _init_playwright_async(self):
        """Initialize Playwright and process actions in this thread (thread-safe)."""
        try:
            print("[Browser] Starting Playwright...")
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=HEADLESS_MODE, slow_mo=SLOW_MO, channel="msedge",
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
            )
            self.context = self.browser.new_context(
                storage_state=AUTH_FILE if os.path.exists(AUTH_FILE) else None
            )
            self.page = self.context.new_page()
            self.page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=30000)
            self.browser_ready = True
            print("[Browser] Ready - Google.com loaded")
            
            # ===== PLAYWRIGHT ACTION LOOP (runs in this thread) =====
            while self.running:
                try:
                    # Check for actions to execute
                    action = self.action_queue.get(timeout=0.1)
                    if action:
                        print(f"[Browser] Executing: {action}")
                        try:
                            execute_action(self.page, action)
                            print(f"[Browser] ✓ Completed: {action}")
                        except Exception as e:
                            print(f"[Browser] Action error: {e}")
                except Empty:
                    pass  # No action, continue loop
                except Exception as e:
                    if self.running:
                        print(f"[Browser] Loop error: {e}")
            
            print("[Browser] Action loop stopped")
            
        except Exception as e:
            print(f"[Browser] Error: {e}")
            self.page = None
    
    def stop(self):
        """Stop the controller."""
        self.running = False
        self.stop_event.set()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        
        self.status.config(text="Status: Stopped")
        self.mode_label.config(text="MODE: IDLE")
    
    def _extract_landmarks(self, hand_landmarks):
        if not hand_landmarks:
            return None
        lm = hand_landmarks[0]
        return np.array([[p.x, p.y, p.z] for p in lm])
    
    def _calc_velocity(self, current, prev):
        if current is None or prev is None:
            return 0.0
        indices = [0, 4, 8]
        max_vel = 0.0
        for i in indices:
            disp = np.sqrt(np.sum((current[i] - prev[i]) ** 2))
            max_vel = max(max_vel, disp)
        return max_vel
    
    def _normalize_window(self, window):
        N = window.shape[0]
        landmarks = window.reshape(N, 21, 3)
        ref_wrist = landmarks[0, 0]
        frame0_centered = landmarks[0] - ref_wrist
        ref_scale = np.abs(frame0_centered).max() + 1e-6
        normalized = (landmarks - ref_wrist) / ref_scale
        return normalized.reshape(N, 63)
    
    def _predict_dynamic(self, seq):
        try:
            if seq.ndim == 3:
                seq = seq.reshape(seq.shape[0], -1)
            seq_len = seq.shape[0]
            if seq_len >= NUM_FRAMES:
                indices = np.linspace(0, seq_len - 1, NUM_FRAMES).astype(int)
                sampled = seq[indices]
            else:
                padding = np.tile(seq[-1], (NUM_FRAMES - seq_len, 1))
                sampled = np.vstack((seq, padding))
            normalized = self._normalize_window(sampled)
            x = torch.from_numpy(normalized.reshape(1, NUM_FRAMES, -1).astype("float32")).to(self.device)
            with torch.no_grad():
                logits = self.dynamic_model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            return {DYNAMIC_GESTURE_CLASSES[i]: float(p) for i, p in enumerate(probs)}
        except:
            return None
    
    def _draw_hand_skeleton(self, frame, landmarks, w, h):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
        ]
        for start, end in connections:
            x1, y1 = int(landmarks[start][0] * w), int(landmarks[start][1] * h)
            x2, y2 = int(landmarks[end][0] * w), int(landmarks[end][1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (56, 189, 248), 2)
        for lm in landmarks:
            x, y = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(frame, (x, y), 4, (248, 113, 113), -1)
    
    def _draw_overlay(self, frame, static_gesture, static_score, dynamic_gesture, dynamic_conf, velocity, gate_open, active_mode):
        h, w = frame.shape[:2]
        
        # Mode indicator
        mode_color = (34, 197, 94) if active_mode == "STATIC" else ((249, 115, 22) if active_mode == "DYNAMIC" else (59, 130, 246))
        cv2.rectangle(frame, (10, 10), (220, 45), mode_color, -1)
        cv2.putText(frame, f"MODE: {active_mode}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Gesture info box
        cv2.rectangle(frame, (10, h - 100), (350, h - 10), (30, 30, 30), -1)
        cv2.putText(frame, f"Static: {static_gesture} ({static_score:.2f})", (15, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 1)
        cv2.putText(frame, f"Dynamic: {dynamic_gesture} ({dynamic_conf:.2f})", (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (249, 115, 22), 1)
        
        gate_text = "GATE: OPEN ✓" if gate_open else "GATE: CLOSED"
        cv2.putText(frame, f"Vel: {velocity:.4f} | {gate_text}", (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (148, 163, 184), 1)
    
    def _process_action_queue(self):
        """Legacy method - actions now processed by Playwright thread."""
        # Actions are now processed by _init_playwright_async loop
        # This method is kept for backwards compatibility
        self.action_busy = False
    
    def update_frame(self):
        """Main processing loop."""
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # FPS
        curr_time = time.time()
        fps = int(1 / max(curr_time - self.prev_time, 1e-6))
        self.prev_time = curr_time
        
        # Initialize frame state
        static_gesture = "NONE"
        static_score = 0.0
        dynamic_gesture = "none"
        dynamic_conf = 0.0
        velocity = 0.0
        gate_open = False
        active_mode = "STATIC"
        
        self.timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # ===== STATIC DETECTION =====
        hand_result = self.hand_landmarker.detect_for_video(mp_image, self.timestamp)
        current_landmarks = self._extract_landmarks(hand_result.hand_landmarks)
        
        if hand_result.hand_landmarks:
            lm = hand_result.hand_landmarks[0]
            thumb, index = lm[4], lm[8]
            pinch_dist = np.linalg.norm(np.array([thumb.x, thumb.y]) - np.array([index.x, index.y]))
            if pinch_dist < PINCH_THRESHOLD:
                static_gesture = "PINCH"
                static_score = 1.0
        
        if static_gesture == "NONE":
            gesture_result = self.gesture_recognizer.recognize_for_video(mp_image, self.timestamp)
            if gesture_result.gestures:
                top = gesture_result.gestures[0][0]
                if top.category_name in ALLOWED_STATIC_GESTURES and top.score >= MIN_STATIC_CONFIDENCE:
                    static_gesture = top.category_name
                    static_score = top.score
        
        # Output static gesture (with 3-second cooldown)
        if static_gesture != "NONE":
            norm_score = normalize_gesture_score(static_score)
            # Only output if cooldown has passed
            if norm_score > 0 and (curr_time - self.last_static_gesture_time) >= STATIC_COOLDOWN:
                self.gesture_result = (static_gesture, norm_score)
                self.last_gesture_time = curr_time
                self.last_static_gesture_time = curr_time  # Update cooldown timer
                self.vote_buffer.clear()
                print(f"[Static] {static_gesture} (cooldown reset)")
        
        # ===== DYNAMIC DETECTION =====
        if static_gesture == "NONE" and current_landmarks is not None:
            active_mode = "DYNAMIC"
            velocity = self._calc_velocity(current_landmarks, self.prev_landmarks)
            gate_open = velocity >= VELOCITY_THRESHOLD
            self.landmark_buffer.append(current_landmarks)
            
            if gate_open and (curr_time - self.last_detection_time > COOLDOWN):
                if len(self.landmark_buffer) >= 5:
                    seq = np.stack(list(self.landmark_buffer), axis=0)
                    preds = self._predict_dynamic(seq)
                    if preds:
                        g_name, conf = max(preds.items(), key=lambda x: x[1])
                        dynamic_gesture = g_name
                        dynamic_conf = conf
                        if conf >= DYNAMIC_CONFIDENCE_THRESHOLD and g_name != "none":
                            self.vote_buffer.append(g_name)
            
            self.prev_landmarks = current_landmarks
            
            if len(self.vote_buffer) >= 10:
                counts = Counter(self.vote_buffer)
                if counts:
                    top_gesture, top_count = counts.most_common(1)[0]
                    if top_count >= VOTE_CONSENSUS and top_gesture != "none":
                        self.confirmed_dynamic_gesture = top_gesture
                        self.confirmed_dynamic_time = curr_time
                        norm_score = normalize_gesture_score(0.95)
                        self.gesture_result = (top_gesture, norm_score)
                        self.last_gesture_time = curr_time
                        self.last_detection_time = curr_time
                        self.vote_buffer.clear()
        else:
            self.prev_landmarks = current_landmarks
        
        # ===== CHECK VOICE QUEUE =====
        try:
            while True:
                modality, action, score = self.result_queue.get_nowait()
                if modality == "voice":
                    self.voice_result = (action, score)
                    self.last_voice_time = curr_time
                    self.voice_label.config(text=f"VOICE: {action} ({score:.2f})")
        except Empty:
            pass
        
        # Clear stale results
        if curr_time - self.last_gesture_time > 0.5:
            self.gesture_result = None
        if curr_time - self.last_voice_time > 0.5:
            self.voice_result = None
            self.voice_label.config(text="VOICE: NONE")
        
        # ===== GATED FUSION (Non-blocking) =====
        if self.gesture_result or self.voice_result:
            winner_mod, winner_action, winner_score = gated_max_selection(self.gesture_result, self.voice_result)
            
            if winner_mod:
                active_mode = winner_mod.upper()
                if winner_mod == "gesture":
                    final_action = intentmap(self.page, winner_action)
                else:
                    final_action = winner_action
                
                # Queue action for non-blocking execution
                if not self.action_busy:
                    self.action_queue.put(final_action)
                    self.last_action = final_action
                    self.last_action_time = curr_time
                    self.action_label.config(text=f"ACTION: {final_action}")
                    # Schedule non-blocking execution
                    self.root.after(1, self._process_action_queue)
                
                self.gesture_result = None
                self.voice_result = None
        
        # Clear action display after 2 seconds
        if self.last_action and (curr_time - self.last_action_time > 2.0):
            self.action_label.config(text="ACTION: ---")
            self.last_action = None
        
        # ===== UI UPDATES =====
        # Dynamic display
        if self.confirmed_dynamic_gesture and (curr_time - self.confirmed_dynamic_time < 2.0):
            dynamic_display = self.confirmed_dynamic_gesture.upper()
            dynamic_conf = 1.0
        else:
            dynamic_display = dynamic_gesture.upper()
            self.confirmed_dynamic_gesture = None
        
        # Mode label
        if static_gesture != "NONE":
            self.mode_label.config(text="MODE: STATIC ●", fg="#22c55e")
        elif self.voice_result:
            self.mode_label.config(text="MODE: VOICE ●", fg="#3b82f6")
        elif gate_open:
            self.mode_label.config(text="MODE: DYNAMIC ●", fg="#f97316")
        else:
            self.mode_label.config(text="MODE: WAITING...", fg="#94a3b8")
        
        # Gesture labels
        self.static_label.config(text=f"STATIC: {static_gesture} ({static_score:.2f})")
        self.dynamic_label.config(text=f"DYNAMIC: {dynamic_display} ({dynamic_conf:.2f})")
        
        gate_status = "OPEN ✓" if gate_open else "CLOSED"
        self.velocity_label.config(text=f"VELOCITY: {velocity:.4f} | GATE: {gate_status}")
        self.fps_label.config(text=f"FPS: {fps}")
        
        # ===== DRAW OVERLAY =====
        if current_landmarks is not None:
            self._draw_hand_skeleton(rgb, current_landmarks, rgb.shape[1], rgb.shape[0])
        self._draw_overlay(rgb, static_gesture, static_score, dynamic_display, dynamic_conf, velocity, gate_open, active_mode)
        
        # Update video display
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)
        
        # Schedule next frame
        self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = MultimodalApp(root)
    root.mainloop()




