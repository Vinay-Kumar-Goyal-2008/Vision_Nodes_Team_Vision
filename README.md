# Hand Gesture Recognition System 🎯

A real-time **hand gesture recognition system** built using **Python**, **OpenCV**, and **MediaPipe Tasks**. This project detects hands from a webcam feed, recognizes gestures, and can be easily integrated into other applications such as virtual controls, HCI systems, automation, or assistive technologies.

---

## ✨ Features

- 📷 Real-time webcam-based hand detection
- ✋ Gesture recognition using MediaPipe Gesture Recognizer
- 🤏 Custom pinch gesture detection (distance-based)
- ⚡ Modular design (can be imported into other Python files)
- 🧠 Confidence filtering for reliable predictions
- 🖥️ Optional GUI support using Tkinter

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **OpenCV** – video capture & visualization
- **MediaPipe Tasks** – hand landmarking & gesture recognition
- **NumPy** – numerical computations
- **Pillow (PIL)** – image handling
- **Tkinter** – GUI (optional)

---

## 📂 Project Structure

```
├── gesture_recognizer.task      # MediaPipe gesture model
├── hand_landmarker.task         # MediaPipe hand landmark model
├── main.py                      # Main execution file
├── gesture_module.py            # Reusable gesture recognition function
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download MediaPipe models**
- `gesture_recognizer.task`
- `hand_landmarker.task`

Place them in the project root directory.

---

## ▶️ Usage

### Run the application
```bash
python main.py
```

### Use as a module in another file
```python
from gesture_module import static_function

result = static_function()
print(result)
```

---

## 🧪 Gesture Logic

- Uses **hand landmarks** to track finger positions
- Pinch gesture detected using **Euclidean distance** between thumb and index finger
- Gesture confidence threshold applied to reduce noise

---

## 📸 Sample Output

- Recognized gesture name displayed on screen
- Hand landmarks drawn in real time
- Stable gesture output after confidence filtering

---

## 🚀 Applications

- Virtual mouse / keyboard
- Smart UI control
- Robotics & drone control
- Sign language recognition (extendable)
- AR/VR interaction systems

---

## 🔮 Future Improvements

- Support for multiple hands
- Custom gesture training
- Voice + gesture fusion
- FPS optimization
- Model quantization for edge devices

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch (`feature/new-feature`)
3. Commit your changes
4. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Vansh Singh**  
Passionate about Computer Vision, AI, and Human–Computer Interaction 🚀

---

⭐ If you like this project, don’t forget to star the repository!!

