# Real-Time Hand Proximity Warning System

This project is a real-time computer vision prototype that tracks a user’s hand position from a live camera feed and triggers visual warnings when the hand approaches a virtual object on the screen.

The system uses **classical computer vision techniques** (color segmentation and contour analysis) and runs entirely on the CPU without pose-estimation libraries or cloud services.

## Features
- Real-time hand/fingertip tracking  
- Virtual object rendered on live camera feed  
- Distance-based interaction states: **SAFE / WARNING / DANGER**  
- Clear on-screen alert (**“DANGER DANGER”**) in the danger state  
- ≥ 8 FPS on CPU-only execution  

## Approach
- Segment a **red-colored hand/object** using HSV color segmentation  
- Extract contours and select the largest one as the hand region  
- Estimate the fingertip using the top-most contour point  
- Compute distance to a virtual boundary and classify interaction state  

## Tech Stack
- Python  
- OpenCV  
- NumPy

## Demo 
<!-- Uploading "demo.mp4"... -->

## Run
```bash
pip install opencv-python numpy
python hand.py
