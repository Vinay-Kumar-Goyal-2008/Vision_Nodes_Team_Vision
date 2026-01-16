import speech_recognition as sr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from playwright.sync_api import sync_playwright
import threading
import time
import os
from queue import Queue

# ---------------- CONFIG ----------------
AUTH_FILE = "auth.json"
HEADLESS_MODE = False
SLOW_MO = 1000
SIMILARITY_THRESHOLD = 0.45
# ----------------------------------------

# ---------------- PLAYWRIGHT (MAIN THREAD ONLY) ----------------
p = sync_playwright().start()
browser = p.chromium.launch(
    headless=HEADLESS_MODE,
    slow_mo=SLOW_MO,
    channel="msedge",
    args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
)

context = browser.new_context(
    storage_state=AUTH_FILE if os.path.exists(AUTH_FILE) else None
)
# ---------------------------------------------------------------

# ---------------- SPEECH SETUP ----------------
r = sr.Recognizer()
r.pause_threshold = 0.8
mic = sr.Microphone()

with mic as source:
    r.adjust_for_ambient_noise(source, duration=1)
# ----------------------------------------------

# ---------------- NLP MODEL ----------------
model = SentenceTransformer("./models/all-MiniLM-L6-v2")
# -------------------------------------------

# ---------------- COMMANDS ----------------
commands = {
    "opencommands": [
        "open youtube",
        "open gmail",
        "open wikipedia"
    ]
}
# -------------------------------------------

# ---------------- THREAD-SAFE QUEUE ----------------
command_queue = Queue()
# --------------------------------------------------


# ================= FUNCTION: COMMAND PREDICTION =================
def get_best_command(text):
    best_cmd, best_cat, best_score = None, None, 0.0
    text_emb = model.encode([text])

    for cat, cmd_list in commands.items():
        emb = model.encode(cmd_list)
        scores = cosine_similarity(text_emb, emb)[0]
        idx = np.argmax(scores)

        if scores[idx] > best_score:
            best_score = scores[idx]
            best_cmd = cmd_list[idx]
            best_cat = cat

    if best_score < SIMILARITY_THRESHOLD:
        return None, None

    return best_cmd, best_cat
# =================================================================


# ================= SPEECH LISTENER THREAD =================
def speech_listener():
    print("Listening...")

    while True:
        try:
            with mic as source:
                audio = r.listen(source)

            text = r.recognize_google(audio)
            print("You said:", text)

            cmd, cat = get_best_command(text)
            if cmd:
                command_queue.put(cmd)

        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            pass
# ===========================================================


# ================= START SPEECH THREAD =================
threading.Thread(
    target=speech_listener,
    daemon=True
).start()
# ======================================================


# ================= MAIN LOOP (PLAYWRIGHT EXECUTION) =================
while True:
    if not command_queue.empty():
        command = command_queue.get()
        print("Executing:", command)

        page = context.new_page()

        if command == "open youtube":
            page.goto("https://www.youtube.com")

        elif command == "open gmail":
            page.goto("https://mail.google.com")

        elif command == "open wikipedia":
            page.goto("https://en.wikipedia.org")

    time.sleep(0.1)
# ==================================================================
