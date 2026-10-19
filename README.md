# ✋ HandPilot

Control your computer with hand gestures using just your webcam.
Move the mouse, click, drag, scroll, copy, paste, undo, redo and more, all without touching anything.

Built with **MediaPipe HandLandmarker**, **OpenCV**, **PyAutoGUI** and **CustomTkinter**.

## Features

- One-click **START / STOP** button
- Real-time hand tracking with a live camera preview
- Cursor control, left click, right click, drag / select, scroll
- Keyboard shortcuts by gesture: copy, paste, select all, undo, redo, back, forward
- Hold-to-confirm for commands (0.6 s) so nothing triggers by accident
- Adjustable smoothness and active area
- Built-in **Gesture Guide** tab
- Works with the latest MediaPipe (Tasks API) on modern Python versions

## Gesture guide

| Gesture | How | Action |
|---|---|---|
| ☝️ Move | Only index finger up | Cursor follows fingertip |
| 🤏 Left click | Quick pinch, thumb + index | Click |
| 🤏 Drag / select | Pinch thumb + index and hold, then move | Press and drag, release to drop |
| 🤌 Right click | Pinch thumb + middle | Right click |
| ✌️ Scroll | Index + middle up, move hand up / down | Scroll |
| 🤟 Copy | Index + middle + ring up (hold) | `Ctrl/Cmd + C` |
| 🖖 Paste | Four fingers up, thumb folded (hold) | `Ctrl/Cmd + V` |
| 🖐️ Select all | Open palm (hold) | `Ctrl/Cmd + A` |
| 👍 Undo | Thumb only (hold) | `Ctrl/Cmd + Z` |
| 🤙 Redo | Thumb + pinky (hold) | `Ctrl + Y` (`Cmd + Shift + Z` on Mac) |
| 🤘 Back | Index + pinky (hold) | `Alt + Left` |
| 🤟 Forward | Thumb + index + pinky (hold) | `Alt + Right` |
| ✊ Rest | Closed fist | Nothing happens (pause) |

"Hold" means keep the gesture steady for about 0.6 seconds. The progress bar in the sidebar shows when it fires.

## Run from source

Requirements: Python 3.9 or newer, and a webcam.

```bash
git clone https://github.com/<your-username>/handpilot.git
cd handpilot

python -m venv venv
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
python hand_mouse_app.py
```

On the first START, the app downloads the hand model (`hand_landmarker.task`, about 7 MB) to:

- Windows: `%APPDATA%\HandPilot`
- Mac / Linux: `~/.handpilot`

You need internet only for that first run. If the download is blocked, save the file from
[this link](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
into that folder manually.

## Build a desktop app (.exe)

```bash
pip install pyinstaller
python build.py
```

The app is created in `dist/HandPilot/`. Run `HandPilot.exe` (Windows) or `HandPilot` (Mac / Linux).
To share it, zip the whole `dist/HandPilot` folder.

Optional: put an `icon.ico` in the project folder before building to give the app an icon.

Build on the same operating system you want to run the app on.

## Tips

- Use even lighting and keep your palm facing the camera.
- Keep your hand inside the orange rectangle; it maps to your whole screen.
- Make a fist to pause the cursor at any time.
- Cursor jittery? Raise **Smoothness**. Reaching too far? Increase **Active area**.

## Troubleshooting

| Problem | Fix |
|---|---|
| Camera does not open | Close Zoom, Teams or other apps using the camera. Check camera permissions. |
| Mac: nothing is controlled | System Settings → Privacy & Security → allow **Camera** and **Accessibility** for the app or terminal. |
| `pip install` can't find mediapipe | Update pip (`python -m pip install --upgrade pip`) and use a 64-bit Python. |
| Model download fails | Download it manually (link above) into the folder shown in the error message. |
| Built exe flagged by antivirus | Common false positive with PyInstaller apps. Build it yourself from source. |
| Gestures misdetected | Improve lighting, keep the whole hand visible, and avoid a busy background. |

## Project structure

```
handpilot/
├── hand_mouse_app.py   # app: gesture engine + UI
├── build.py            # PyInstaller build script
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: HandPilot"
git branch -M main
git remote add origin https://github.com/<your-username>/handpilot.git
git push -u origin main
```

Upload the zipped `dist/HandPilot` folder as a **Release** so others can download the app without installing Python.

## License

MIT, see [LICENSE](LICENSE).
