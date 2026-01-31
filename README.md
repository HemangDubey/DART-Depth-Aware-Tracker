<p align="center">
  <img src="https://img.shields.io/badge/D.A.R.T.-Depth%20Aware%20Real--time%20Tracker-00d4ff?style=for-the-badge&logo=target&logoColor=white" alt="D.A.R.T."/>
</p>

<h1 align="center">🎯 D.A.R.T. - Depth-Aware Real-time Tracker</h1>

<p align="center">
  <strong>Real-time Object Detection + Depth Estimation + WebSocket Streaming</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?style=flat-square&logo=yolo&logoColor=black"/>
  <img src="https://img.shields.io/badge/SCDepthV3-Depth%20Estimation-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/WebSocket-Real--time-blue?style=flat-square&logo=socket.io&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=flat-square&logo=googlecolab&logoColor=white"/>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-google-colab-setup">Colab Setup</a> •
  <a href="#-local-setup">Local Setup</a> •
  <a href="#-architecture">Architecture</a>
</p>

---

## 🎬 Demo

<p align="center">
  <a href="https://drive.google.com/file/d/15al5A8g_k1x9HH_Q5H1rN1qm0O5I9HBl/view?usp=sharing">
    <img src="https://img.shields.io/badge/▶️%20Watch%20Demo-Implementation%20Video-red?style=for-the-badge&logo=googledrive&logoColor=white" alt="Watch Demo"/>
  </a>
</p>

### 📸 Live Dashboard Preview

| Real-time Detection & Tracking | Bird's Eye View (BEV) |
|:---:|:---:|
| ![Detection](https://via.placeholder.com/400x250/1a1a2e/00d4ff?text=YOLO+Detection+%2B+Tracking) | ![BEV](https://via.placeholder.com/400x250/1a1a2e/00ff88?text=Bird%27s+Eye+View+Map) |
| *Objects tracked with unique IDs and depth-aware color coding* | *Spatial mapping showing Close/Medium/Far zones* |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Real-time Object Detection** | YOLOv11 detects vehicles, pedestrians, traffic lights, and more |
| 📏 **Depth Estimation** | SCDepthV3 provides metric depth for each detected object |
| 🔄 **Object Tracking** | Persistent IDs track objects across frames with motion trails |
| 🗺️ **Bird's Eye View (BEV)** | Top-down tactical view with Close/Medium/Far zone classification |
| 🌐 **WebSocket Streaming** | Real-time video stream to browser dashboard |
| ☁️ **Google Colab Support** | Run on free GPU with Ngrok tunneling |
| 💻 **Local Execution** | Run entirely on your machine with CUDA support |
| 🎨 **Beautiful Dashboard** | Modern, responsive web interface with live FPS counter |

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended - Free GPU)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HemangDubey/DART-Depth-Aware-Tracker/blob/main/CATERPILLAR_AI_MODEL.ipynb)

### Option 2: Local Machine
```bash
git clone https://github.com/HemangDubey/DART-Depth-Aware-Tracker.git
cd DART-Depth-Aware-Tracker
pip install -r requirements.txt
python run_local.py
```

---

## ☁️ Google Colab Setup

This is the **recommended method** as it provides free GPU acceleration and doesn't require any local setup.

### Prerequisites
- Google Account
- Ngrok Account (free tier works) - [Sign up here](https://ngrok.com/)

### Step-by-Step Guide

#### Step 1: Get Ngrok Auth Token
1. Go to [Ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)
2. Copy your authtoken (looks like: `2abc123xyz...`)

#### Step 2: Open the Notebook
Click the button below to open in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HemangDubey/DART-Depth-Aware-Tracker/blob/main/CATERPILLAR_AI_MODEL.ipynb)

#### Step 3: Run All Cells in Order

| Cell | Purpose | What to Do |
|------|---------|------------|
| **Cell 1** | Install Dependencies | Just run it, wait ~2 mins |
| **Cell 2** | Mount Google Drive | Click "Connect" when prompted |
| **Cell 3** | Download Models | Downloads YOLO & SCDepthV3 (~500MB) |
| **Cell 4** | Load Models | Initializes AI models on GPU |
| **Cell 5** | Setup Ngrok | **⚠️ Paste your Ngrok token here** |
| **Cell 6** | Define WebSocket Server | Sets up the streaming server |
| **Cell 7** | Start Server | **🚀 Server starts here!** |

#### Step 4: Connect the Dashboard
1. After Cell 7 runs, you'll see:
   ```
   === WS Server Ready ===
   === Connect client to: wss://xxxx-xxxx.ngrok-free.dev ===
   ```
2. Copy the `wss://...` URL
3. Open `websocket.html` in your browser
4. Paste the URL and click **Connect**

#### Step 5: Watch the Magic! 🎉
- Live video stream with object detection
- Depth-aware tracking with color-coded boxes
- Bird's Eye View showing spatial positions

---

## 💻 Local Setup

Run D.A.R.T. entirely on your local machine with CUDA GPU acceleration.

### System Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for models

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/HemangDubey/DART-Depth-Aware-Tracker.git
cd DART-Depth-Aware-Tracker
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Download Models
The models will be downloaded automatically on first run, or you can download manually:

| Model | Size | Download |
|-------|------|----------|
| YOLOv11n | ~6MB | Auto-downloads via Ultralytics |
| SCDepthV3 | ~100MB | Auto-downloads from HuggingFace |

#### 5. Add Your Video
Place your input video in the project folder and update the path in `run_local.py`:
```python
VIDEO_PATH = "your_video.mp4"
```

#### 6. Run the Application
```bash
python run_local.py
```

#### 7. Open the Dashboard
1. Open `websocket.html` in Chrome/Firefox
2. Enter: `ws://localhost:8765`
3. Click **Connect**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        D.A.R.T. System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │              │    │         AI Processing Pipeline       │  │
│  │  Video       │───▶│  ┌─────────┐  ┌─────────────────┐   │  │
│  │  Source      │    │  │ YOLOv11 │  │   SCDepthV3     │   │  │
│  │  (MP4/RTSP)  │    │  │ Detect  │  │   Depth Est.    │   │  │
│  │              │    │  └────┬────┘  └────────┬────────┘   │  │
│  └──────────────┘    │       │                │            │  │
│                      │       ▼                ▼            │  │
│                      │  ┌──────────────────────────────┐   │  │
│                      │  │      Fusion + Tracking       │   │  │
│                      │  │  • Object-Depth Association  │   │  │
│                      │  │  • BoT-SORT Tracking         │   │  │
│                      │  │  • BEV Mapping               │   │  │
│                      │  └──────────────┬───────────────┘   │  │
│                      └─────────────────┼───────────────────┘  │
│                                        │                      │
│                                        ▼                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                 WebSocket Server                          │ │
│  │  • Encodes frames as JPEG                                │ │
│  │  • Streams via ws:// or wss:// (Ngrok)                   │ │
│  └──────────────────────────┬───────────────────────────────┘ │
│                             │                                 │
└─────────────────────────────┼─────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Web Dashboard (Browser)                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Real-time video display                              │   │
│  │  • Connection status & FPS counter                      │   │
│  │  • Server logs                                          │   │
│  │  • Fullscreen mode                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
DART-Depth-Aware-Tracker/
├── 📄 CATERPILLAR_AI_MODEL.ipynb  # Google Colab notebook
├── 📄 run_local.py                 # Local execution script
├── 📄 websocket.html               # Web dashboard
├── 📄 SC_DepthV3.py                # Depth estimation module
├── 📄 requirements.txt             # Python dependencies
├── 📁 Python-Depth-Est-AV2/        # SCDepthV3 model code
│   └── ckpts/                      # Model checkpoints (auto-download)
└── 📄 README.md                    # You are here!
```

---

## 🎨 Dashboard Features

### Controls Panel
- **WebSocket URL Input**: Enter your server URL
- **Connect/Disconnect**: Manage connection
- **Status Indicator**: Green (connected), Yellow (connecting), Gray (disconnected)
- **FPS Counter**: Real-time frame rate display

### Video Display
- **Live Stream**: Full detection visualization
- **Bounding Boxes**: Color-coded by distance
  - 🟢 Green = Far (safe)
  - 🟡 Yellow = Medium (caution)
  - 🔴 Red = Close (alert)
- **Tracking IDs**: Persistent object identification
- **BEV Panel**: Bird's eye view with zone classification

### Fullscreen Mode
Click the fullscreen button for immersive monitoring.

---

## 🛠️ Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **Colab disconnects** | Keep the tab active, use Colab Pro for longer sessions |
| **Ngrok tunnel fails** | Check your authtoken, ensure you're on free tier limits |
| **No video stream** | Verify video file path, check CUDA availability |
| **Low FPS** | Reduce video resolution, use T4 GPU in Colab |
| **WebSocket won't connect** | Check URL format (ws:// or wss://), firewall settings |

### GPU Memory Issues
If you encounter CUDA out of memory:
```python
# Add to your code
import torch
torch.cuda.empty_cache()
```

---

## 🔧 Configuration

### Adjustable Parameters in `run_local.py`

```python
# Video Settings
VIDEO_PATH = "your_video.mp4"
PROCESS_WIDTH = 640       # Processing resolution
PROCESS_HEIGHT = 480

# Detection Settings
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# WebSocket Settings
WEBSOCKET_PORT = 8765

# Depth Zones (in meters)
CLOSE_THRESHOLD = 5.0
MEDIUM_THRESHOLD = 15.0
```

---

## 📚 Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core programming language |
| **PyTorch** | Deep learning framework |
| **Ultralytics YOLOv11** | Object detection model |
| **SCDepthV3** | Monocular depth estimation |
| **OpenCV** | Image processing |
| **WebSockets** | Real-time streaming |
| **Ngrok** | Secure tunneling for Colab |
| **TailwindCSS** | Dashboard styling |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Hemang Dubey**

[![GitHub](https://img.shields.io/badge/GitHub-HemangDubey-181717?style=flat-square&logo=github)](https://github.com/HemangDubey)

---

<p align="center">
  <strong>⭐ If you found this project helpful, please give it a star! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ and ☕
</p>
