# Gesture Recognition with OpenCV & MediaPipe

Inspired by the viral monkey meme, this project uses **MediaPipe** and **OpenCV** to detect and classify hand and face gestures in real time through your webcam.
It can recognize several gestures such as:
- **Thumbs Up**
- **Pointing**
- **Thinking**
- **Neutral**

When a gesture is detected, the corresponding image will appear beside the webcam feed.

---

## Requirements

Make sure you have **Python 3.8+** installed.

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
gesture_recognition/
│
├── gesture-tracker.py                # Main program
├── thumbs_up.jpg          # Image for thumbs up gesture
├── pointing.jpg           # Image for pointing gesture
├── thinking.jpg           # Image for thinking gesture
├── neutral.jpg            # Image for neutral gesture
├── requirements.txt            # Requirements for the program
└── README.md
```

Make sure all image files are placed in the same directory as `gesture-tracker.py`.

---

## How to Run

Run the main script:
```bash
python gesture-tracker.py
```

Controls:
- Press **q** or **ESC** → Quit the program.

---

## How It Works

- **Hand Detection** → Uses `MediaPipe Hands` to detect landmarks and finger positions.  
- **Face Detection** → Uses `MediaPipe FaceMesh` to locate the nose tip.  
- **Gesture Classification** → Determines gesture type based on relative landmark positions.  
- **Image Display** → Loads and resizes the corresponding image, then displays it beside the camera feed.

---

## Customization

You can add more gestures by:
1. Creating a new image file
2. Adding it to the `IMAGE_PATHS` dictionary
3. Implementing logic in `classify_gesture()` to detect that gesture