# Drowsiness Detection System

A real-time drowsiness detection system that monitors eye movements using computer vision techniques to alert drivers when signs of drowsiness are detected.

## Features

- **Real-time facial detection** using YOLOv8
- **Precise eye landmark tracking** with MediaPipe Face Mesh
- **Eye Aspect Ratio (EAR) calculation** to detect drowsiness
- **Customizable alert thresholds** for different sensitivity levels
- **Audio alerts** when drowsiness is detected
- **Visual indicators** on the video feed

## How It Works

This system uses the Eye Aspect Ratio (EAR) to detect drowsiness:

1. **Face Detection**: YOLOv8 model detects faces in the video feed
2. **Landmark Detection**: MediaPipe Face Mesh identifies key points around the eyes
3. **EAR Calculation**: Calculates the ratio of eye height to width
4. **Drowsiness Detection**: When EAR exceeds a threshold for a specified duration, the system triggers an alert

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- NumPy
- MediaPipe
- Windows (for sound alerts)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install opencv-python torch ultralytics numpy mediapipe
   ```

5. Download YOLOv8 weights:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

## Usage

Run the script using:
```bash
python EAR.py
```

- Press 'q' to quit the application
- The system will beep and display a warning when drowsiness is detected

## Customization

You can adjust the following parameters in the code:

- `EAR_THRESHOLD`: Adjust the sensitivity of drowsiness detection (default: 1.50)
- `DURATION_THRESHOLD`: Number of seconds the EAR must exceed the threshold to trigger an alert (default: 5)

## How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [MediaPipe](https://github.com/google/mediapipe) for facial landmark detection
- Research papers on Eye Aspect Ratio for drowsiness detection
