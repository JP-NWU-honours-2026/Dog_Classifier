# CanineVision AI - Dog Breed Classification Web App

A self-contained, real-time web application for the **ITRI626 Dog Classifier Mini-Project**. Allows users to upload custom dog photos or click sample images to test the trained deep learning model (`best_model.pth`).

---

## Features
- **Modern Glassmorphic Dark UI**: Built with pure Vanilla HTML5, CSS3, and JavaScript (no npm or build tools needed).
- **FastAPI Backend**: Sub-50ms CPU inference using PyTorch and torchvision.
- **Interactive Drag & Drop**: Upload custom dog pictures with live preview.
- **1-Click Test Samples**: Preloaded sample Chihuahua and Siberian Husky images.
- **Real-Time Visualizations**: Glowing winner badge, confidence percentages, comparative probability breakdown bars, and inference latency telemetry.

---

## How to Run the App

1. Open your terminal in the workspace root:
   ```powershell
   python Dog_Classifier_App/server.py
   ```

2. Open your web browser and navigate to:
   ```text
   http://localhost:8000
   ```

3. Drag & drop any dog image into the upload box (or click one of the quick test sample buttons) and view the model's prediction!
