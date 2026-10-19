"""
HandPilot - control your computer with hand gestures.

Tech: MediaPipe HandLandmarker (Tasks API) + OpenCV + PyAutoGUI + CustomTkinter
Works with the NEW mediapipe versions (0.10.30+ / 1.x) on any recent Python.

Install:   pip install -r requirements.txt
Run:       python hand_mouse_app.py

The first time you press START, the app downloads the hand model
(hand_landmarker.task, ~7 MB) next to this file.
"""

import math
import os
import platform
import threading
import time
import urllib.request
from collections import Counter, deque

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import customtkinter as ctk
from PIL import Image

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

IS_MAC = platform.system() == "Darwin"
IS_WIN = platform.system() == "Windows"
MOD = "command" if IS_MAC else "ctrl"

SCREEN_W, SCREEN_H = pyautogui.size()
CAM_W, CAM_H = 640, 480

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model():
    """Download the hand model once if it is missing."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    tmp = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "Could not download the hand model (" + str(e) + ").\n"
            "Download it manually from:\n" + MODEL_URL +
            "\nand save it next to hand_mouse_app.py as hand_landmarker.task"
        )


def back():
    pyautogui.hotkey("command", "[") if IS_MAC else pyautogui.hotkey("alt", "left")


def forward():
    pyautogui.hotkey("command", "]") if IS_MAC else pyautogui.hotkey("alt", "right")


# Gestures that fire once after being HELD steadily
COMMANDS = {
    "COPY": lambda: pyautogui.hotkey(MOD, "c"),
    "PASTE": lambda: pyautogui.hotkey(MOD, "v"),
    "SELECT_ALL": lambda: pyautogui.hotkey(MOD, "a"),
    "UNDO": lambda: pyautogui.hotkey(MOD, "z"),
    "REDO": lambda: pyautogui.hotkey(MOD, "shift", "z") if IS_MAC else pyautogui.hotkey("ctrl", "y"),
    "BACK": back,
    "FORWARD": forward,
}

# (thumb, index, middle, ring, pinky) -> gesture name
FINGER_MAP = {
    (0, 1, 0, 0, 0): "MOVE",
    (1, 1, 0, 0, 0): "MOVE",
    (0, 1, 1, 0, 0): "SCROLL",
    (0, 1, 1, 1, 0): "COPY",
    (0, 1, 1, 1, 1): "PASTE",
    (1, 1, 1, 1, 1): "SELECT_ALL",
    (1, 0, 0, 0, 0): "UNDO",
    (1, 0, 0, 0, 1): "REDO",
    (0, 1, 0, 0, 1): "BACK",
    (1, 1, 0, 0, 1): "FORWARD",
    (0, 0, 0, 0, 0): "REST",
}

LABELS = {
    "NONE": "No hand detected",
    "MOVE": "Moving cursor",
    "SCROLL": "Scrolling",
    "PINCH": "Click / Drag",
    "RIGHT_PINCH": "Right click",
    "COPY": "Copy",
    "PASTE": "Paste",
    "SELECT_ALL": "Select all",
    "UNDO": "Undo",
    "REDO": "Redo",
    "BACK": "Back",
    "FORWARD": "Forward",
    "REST": "Resting (paused)",
    "UNKNOWN": "...",
}


# ----------------------------------------------------------------------------
# Gesture engine
# ----------------------------------------------------------------------------
class GestureEngine:
    HOLD = 0.6          # seconds to hold a command gesture
    COOLDOWN = 0.8      # seconds between commands
    DRAG_AFTER = 0.4    # pinch held longer than this = drag / select

    def __init__(self, margin=100, smoothing=5.0):
        ensure_model()
        with open(MODEL_PATH, "rb") as f:
            model_bytes = f.read()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._t_start = time.time()
        self._last_ts = -1

        # user-adjustable
        self.margin = margin
        self.smoothing = smoothing

        # state
        self.gesture = "NONE"
        self.progress = 0.0
        self.hist = deque(maxlen=5)
        self.cur_g = None
        self.g_t0 = 0.0
        self.fired = False
        self.last_cmd = 0.0

        self.prev_x, self.prev_y = pyautogui.position()
        self.pinching = False
        self.pinch_t0 = 0.0
        self.pinch_lost = 0
        self.dragging = False

        self.right_active = False
        self.last_right = 0.0

        self.scroll_prev_y = None
        self.scroll_acc = 0.0
        self.scroll_gain = 25.0 if IS_WIN else 0.25

    # ---- geometry helpers -------------------------------------------------
    @staticmethod
    def _d(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _fingers(self, lm):
        wrist = lm[0]
        up = [1 if self._d(lm[4], lm[17]) > self._d(lm[3], lm[17]) * 1.05 else 0]
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            up.append(1 if self._d(lm[tip], wrist) > self._d(lm[pip], wrist) * 1.05 else 0)
        return tuple(up)

    def _classify(self, lm):
        size = self._d(lm[0], lm[9]) + 1e-6
        if self._d(lm[4], lm[8]) / size < 0.28:
            return "PINCH"
        if self._d(lm[4], lm[12]) / size < 0.28:
            return "RIGHT_PINCH"
        return FINGER_MAP.get(self._fingers(lm), "UNKNOWN")

    # ---- mouse helpers ----------------------------------------------------
    def _move(self, px, py):
        m = self.margin
        x = np.interp(px, (m, CAM_W - m), (0, SCREEN_W))
        y = np.interp(py, (m, CAM_H - m), (0, SCREEN_H))
        x = self.prev_x + (x - self.prev_x) / max(self.smoothing, 1)
        y = self.prev_y + (y - self.prev_y) / max(self.smoothing, 1)
        x = min(max(x, 0), SCREEN_W - 1)
        y = min(max(y, 0), SCREEN_H - 1)
        pyautogui.moveTo(int(x), int(y))
        self.prev_x, self.prev_y = x, y

    def _update_pinch(self, active, now, px, py):
        if active:
            self.pinch_lost = 0
            if not self.pinching:
                self.pinching = True
                self.pinch_t0 = now
            elif not self.dragging and now - self.pinch_t0 > self.DRAG_AFTER:
                pyautogui.mouseDown()
                self.dragging = True
            if self.dragging:
                self._move(px, py)
        elif self.pinching:
            self.pinch_lost += 1
            if self.pinch_lost >= 3:
                if self.dragging:
                    pyautogui.mouseUp()
                elif now - self.pinch_t0 < self.DRAG_AFTER:
                    pyautogui.click()
                self.pinching = False
                self.dragging = False

    def release_all(self):
        if self.dragging:
            pyautogui.mouseUp()
        self.pinching = False
        self.dragging = False
        self.right_active = False
        self.scroll_prev_y = None

    # ---- detection --------------------------------------------------------
    def _detect(self, rgb):
        ts = int((time.time() - self._t_start) * 1000)
        if ts <= self._last_ts:          # timestamps must strictly increase
            ts = self._last_ts + 1
        self._last_ts = ts
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return self.landmarker.detect_for_video(mp_image, ts)

    @staticmethod
    def _draw(frame, lm):
        pts = [(int(p.x * CAM_W), int(p.y * CAM_H)) for p in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (255, 255, 255), -1)
            cv2.circle(frame, p, 4, (40, 40, 40), 1)

    # ---- main entry -------------------------------------------------------
    def process(self, frame_bgr):
        frame = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._detect(rgb)
        now = time.time()

        m = self.margin
        cv2.rectangle(frame, (m, m), (CAM_W - m, CAM_H - m), (255, 140, 0), 2)

        if not result.hand_landmarks:
            self.release_all()
            self.hist.clear()
            self.gesture, self.progress, self.cur_g = "NONE", 0.0, None
            return frame

        lm = result.hand_landmarks[0]
        self._draw(frame, lm)

        self.hist.append(self._classify(lm))
        g = Counter(self.hist).most_common(1)[0][0]
        self.gesture = g

        ix, iy = lm[8].x * CAM_W, lm[8].y * CAM_H
        mx, my = (lm[4].x + lm[8].x) / 2 * CAM_W, (lm[4].y + lm[8].y) / 2 * CAM_H

        # --- pinch (click / drag-select) ---
        self._update_pinch(g == "PINCH", now, mx, my)

        # --- right click (edge triggered) ---
        if g == "RIGHT_PINCH":
            if not self.right_active and now - self.last_right > 0.8:
                pyautogui.rightClick()
                self.last_right = now
            self.right_active = True
        else:
            self.right_active = False

        # --- move ---
        if g == "MOVE":
            self._move(ix, iy)

        # --- scroll ---
        if g == "SCROLL":
            if self.scroll_prev_y is not None:
                self.scroll_acc += (self.scroll_prev_y - iy) * self.scroll_gain
                n = int(self.scroll_acc)
                if n:
                    pyautogui.scroll(n)
                    self.scroll_acc -= n
            self.scroll_prev_y = iy
        else:
            self.scroll_prev_y = None
            self.scroll_acc = 0.0

        # --- hold-to-trigger commands ---
        if g != self.cur_g:
            self.cur_g, self.g_t0, self.fired = g, now, False
        self.progress = 0.0
        if g in COMMANDS:
            self.progress = min((now - self.g_t0) / self.HOLD, 1.0)
            if self.progress >= 1.0 and not self.fired and now - self.last_cmd > self.COOLDOWN:
                COMMANDS[g]()
                self.fired = True
                self.last_cmd = now

        return frame

    def close(self):
        self.release_all()
        try:
            self.landmarker.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Gesture guide content
# ----------------------------------------------------------------------------
GUIDE = [
    ("☝️", "Move cursor", "Only your index finger up", "Cursor follows your fingertip", "Move"),
    ("🤏", "Left click", "Quickly pinch thumb + index together", "Single left click", "Click"),
    ("🤏", "Drag / Select text", "Pinch thumb + index and HOLD, then move", "Press & drag (select text, move items). Release to let go", "Select"),
    ("🤌", "Right click", "Pinch thumb + middle finger", "Opens context menu", "Click"),
    ("✌️", "Scroll", "Index + middle finger up, move hand up / down", "Scrolls the page", "Scroll"),
    ("🤟", "Copy", "Index + middle + ring finger up (hold 0.6 s)", "Ctrl/Cmd + C", "Command"),
    ("🖖", "Paste", "Four fingers up, thumb folded (hold 0.6 s)", "Ctrl/Cmd + V", "Command"),
    ("🖐️", "Select all", "Open palm, all 5 fingers (hold 0.6 s)", "Ctrl/Cmd + A", "Command"),
    ("👍", "Undo", "Thumb only up, other fingers folded (hold 0.6 s)", "Ctrl/Cmd + Z", "Command"),
    ("🤙", "Redo", "Thumb + pinky out (hold 0.6 s)", "Ctrl + Y  (Cmd+Shift+Z on Mac)", "Command"),
    ("🤘", "Back", "Index + pinky up (hold 0.6 s)", "Alt + Left (browser / folder back)", "Command"),
    ("🤟", "Forward", "Thumb + index + pinky (hold 0.6 s)", "Alt + Right", "Command"),
    ("✊", "Rest / Pause", "Closed fist", "Nothing happens - use it to relax your hand", "Safe"),
]

TIPS = [
    "Use good, even lighting and keep your palm facing the camera.",
    "Keep your hand inside the orange rectangle - it maps to your whole screen.",
    "Command gestures need a 0.6 s steady hold; watch the progress bar fill.",
    "Make a fist (REST) whenever you want to stop the cursor without closing the app.",
    "Tune 'Smoothness' and 'Active area' in the sidebar until it feels natural.",
]

TAG_COLORS = {
    "Move": "#3b82f6", "Click": "#22c55e", "Select": "#a855f7",
    "Scroll": "#06b6d4", "Command": "#f59e0b", "Safe": "#64748b",
}


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HandPilot - Gesture Mouse Control")
        self.geometry("1100x700")
        self.minsize(980, 640)

        self.engine = None
        self.thread = None
        self.running = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.info = ""
        self.fps = 0.0
        self.smooth_val = 5.0
        self.area_val = 100

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(30, self._ui_loop)

    # ---- layout -----------------------------------------------------------
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=270, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsw")
        sb.grid_propagate(False)

        ctk.CTkLabel(sb, text="✋ HandPilot", font=ctk.CTkFont(size=26, weight="bold")).pack(pady=(28, 2))
        ctk.CTkLabel(sb, text="Control your PC with your hand", text_color="gray70").pack(pady=(0, 24))

        self.btn = ctk.CTkButton(
            sb, text="▶  START", height=52, font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#16a34a", hover_color="#15803d", command=self.toggle,
        )
        self.btn.pack(padx=24, fill="x")

        self.status = ctk.CTkLabel(sb, text="● Stopped", text_color="#ef4444", font=ctk.CTkFont(size=14, weight="bold"))
        self.status.pack(pady=(14, 22))

        ctk.CTkLabel(sb, text="Current gesture", text_color="gray70").pack(anchor="w", padx=24)
        self.gesture_lbl = ctk.CTkLabel(sb, text="-", font=ctk.CTkFont(size=20, weight="bold"))
        self.gesture_lbl.pack(anchor="w", padx=24, pady=(2, 6))
        self.prog = ctk.CTkProgressBar(sb, height=10)
        self.prog.set(0)
        self.prog.pack(padx=24, fill="x")
        self.fps_lbl = ctk.CTkLabel(sb, text="FPS: -", text_color="gray60")
        self.fps_lbl.pack(anchor="w", padx=24, pady=(6, 24))

        ctk.CTkLabel(sb, text="Smoothness").pack(anchor="w", padx=24)
        self.smooth = ctk.CTkSlider(sb, from_=1, to=12, number_of_steps=11, command=self._on_settings)
        self.smooth.set(5)
        self.smooth.pack(padx=24, fill="x", pady=(2, 16))

        ctk.CTkLabel(sb, text="Active area (larger box = need smaller moves)").pack(anchor="w", padx=24)
        self.area = ctk.CTkSlider(sb, from_=40, to=180, number_of_steps=14, command=self._on_settings)
        self.area.set(100)
        self.area.pack(padx=24, fill="x", pady=(2, 16))

        self.preview_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(sb, text="Show camera preview", variable=self.preview_var).pack(anchor="w", padx=24, pady=8)

        ctk.CTkLabel(
            sb, text="Tip: make a fist ✊ to pause\nthe cursor any time.",
            text_color="gray60", justify="left",
        ).pack(side="bottom", anchor="w", padx=24, pady=20)

    def _build_main(self):
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        cam_tab = self.tabs.add("📷  Camera")
        guide_tab = self.tabs.add("📖  Gesture Guide")

        self.cam_lbl = ctk.CTkLabel(
            cam_tab, text="Press START to open the camera",
            font=ctk.CTkFont(size=16), text_color="gray60",
        )
        self.cam_lbl.pack(expand=True, fill="both")

        scroll = ctk.CTkScrollableFrame(guide_tab)
        scroll.pack(expand=True, fill="both")
        ctk.CTkLabel(scroll, text="Hand gesture guide", font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=8, pady=(4, 10))

        for emoji, name, how, action, tag in GUIDE:
            card = ctk.CTkFrame(scroll, corner_radius=12)
            card.pack(fill="x", padx=6, pady=5)
            card.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(card, text=emoji, font=ctk.CTkFont(size=36), width=70).grid(row=0, column=0, rowspan=3, padx=10, pady=8)
            ctk.CTkLabel(card, text=name, font=ctk.CTkFont(size=16, weight="bold"), anchor="w").grid(row=0, column=1, sticky="w", pady=(8, 0))
            ctk.CTkLabel(card, text="How:  " + how, anchor="w", text_color="gray80", wraplength=560, justify="left").grid(row=1, column=1, sticky="w")
            ctk.CTkLabel(card, text="Does:  " + action, anchor="w", text_color="gray60", wraplength=560, justify="left").grid(row=2, column=1, sticky="w", pady=(0, 8))
            ctk.CTkLabel(
                card, text=tag, fg_color=TAG_COLORS[tag], corner_radius=8,
                width=80, text_color="white", font=ctk.CTkFont(size=12, weight="bold"),
            ).grid(row=0, column=2, rowspan=3, padx=14)

        ctk.CTkLabel(scroll, text="Tips", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", padx=8, pady=(16, 4))
        for t in TIPS:
            ctk.CTkLabel(scroll, text="•  " + t, anchor="w", justify="left", wraplength=700, text_color="gray80").pack(anchor="w", padx=14, pady=2)

    # ---- settings ---------------------------------------------------------
    def _on_settings(self, _=None):
        self.smooth_val = float(self.smooth.get())
        self.area_val = int(self.area.get())
        eng = self.engine
        if eng:
            eng.smoothing = self.smooth_val
            eng.margin = self.area_val

    # ---- start / stop -----------------------------------------------------
    def toggle(self):
        self.stop() if self.running else self.start()

    def start(self):
        self.error = None
        self.info = "Starting... (first run downloads the hand model)"
        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.btn.configure(text="■  STOP", fg_color="#dc2626", hover_color="#b91c1c")
        self.status.configure(text="● Running", text_color="#22c55e")
        self.tabs.set("📷  Camera")

    def stop(self):
        self.running = False
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        self.thread = None
        self.latest = None
        self.cam_lbl.configure(image=None, text="Press START to open the camera")
        self.cam_lbl.image = None
        self.btn.configure(text="▶  START", fg_color="#16a34a", hover_color="#15803d")
        self.status.configure(text="● Stopped", text_color="#ef4444")
        self.gesture_lbl.configure(text="-")
        self.prog.set(0)
        self.fps_lbl.configure(text="FPS: -")

    # ---- camera thread ----------------------------------------------------
    def _worker(self):
        cap = None
        engine = None
        try:
            self.info = "Loading hand model..."
            engine = GestureEngine(margin=self.area_val, smoothing=self.smooth_val)

            self.info = "Opening camera..."
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if IS_WIN else cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            if not cap.isOpened():
                raise RuntimeError("Could not open the camera. Close other apps using it and check permissions.")

            self.engine = engine
            last = time.time()
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.resize(frame, (CAM_W, CAM_H))
                try:
                    out = engine.process(frame)
                except Exception as e:
                    print("frame error:", e)
                    continue
                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 * (1 / max(now - last, 1e-3))
                last = now
                with self.lock:
                    self.latest = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.error = str(e)
        finally:
            self.engine = None
            if cap is not None:
                cap.release()
            if engine is not None:
                engine.close()

    # ---- UI refresh -------------------------------------------------------
    def _ui_loop(self):
        if self.error:
            msg, self.error = self.error, None
            self.stop()
            self.cam_lbl.configure(text="⚠ " + msg)
        elif self.running:
            eng = self.engine
            if eng is None:
                self.cam_lbl.configure(image=None, text=self.info)
            else:
                with self.lock:
                    img = self.latest
                if self.preview_var.get():
                    if img is not None:
                        ctk_img = ctk.CTkImage(Image.fromarray(img), size=(CAM_W, CAM_H))
                        self.cam_lbl.configure(image=ctk_img, text="")
                        self.cam_lbl.image = ctk_img
                else:
                    self.cam_lbl.configure(image=None, text="Preview hidden (still tracking)")
                self.gesture_lbl.configure(text=LABELS.get(eng.gesture, eng.gesture))
                self.prog.set(eng.progress)
                self.fps_lbl.configure(text=f"FPS: {self.fps:.0f}")
        self.after(25, self._ui_loop)

    def on_close(self):
        if self.running:
            self.stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()