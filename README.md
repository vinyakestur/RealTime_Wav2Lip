# Wav2Lip: Real-Time Lip Sync on iOS using CoreML

This project brings the powerful Wav2Lip model to iOS, enabling real-time lip synchronization between a given audio track and a static face image. Built using Swift, CoreML, and AVFoundation, the app demonstrates advanced deep learning, signal processing, and UI integration in a mobile environment.


## Team Members
- Yashas Besanahalli Vasudeva
- Vibha Kestur Tumakuru Arun Kumar
- Vinya Kestur Tumakuru Arun Kumar
- Vrushank Ramagondanahalli Prasanna Kumar

##  Features
- **Real-time audio capture and processing**
- **Mel spectrogram generation using vDSP**
- **Face detection and masking using Vision framework**
- **CoreML inference for lip generation**
- **Output video rendering with synced lip movements**
- **Gradient-based overlay for realistic output**

##  Tech Stack
- **Language**: Swift
- **Frameworks**: CoreML, AVFoundation, Vision, SwiftUI
- **Signal Processing**: Accelerate (vDSP)
- **Model**: Quantized Wav2Lip CoreML format
- **Input Shapes**:
  - Audio (mel spectrogram): `[5, 1, 80, 16]`
  - Face frames: `[5, 3, 96, 96]`

## Model Workflow

1. Audio Processing:
   - Captured via AVAudioEngine.
   - Converted to mel spectrograms using STFT + Hanning window via vDSP.

2. Face Processing:
   - Face detected via Vision framework.
   - Cropped and masked (lower half).
   - Resized and duplicated across 5 frames.

3. CoreML Inference:
   - Inputs: Spectrogram + Face sequence.
   - Output: Lip-synced frames (`[5, 3, 96, 96]`).

4. Video Generation:
   - Frames are compiled into the final video using `VideoGenerator.swift`.

## Core Swift Files

| File | Responsibility |
|------|----------------|
| `LipSyncProcessor.swift` | Manages model inference |
| `AudioProcessor.swift` | Converts audio to mel spectrogram |
| `AudioManager.swift` | Real-time audio capture |
| `FaceProcessor.swift` | Detects, crops, and masks face region |
| `VideoGenerator.swift` | Compiles generated frames into video |
| `ContentView.swift` | UI layer for interaction |

##  Challenges and Solutions
- Latency Issues: Solved using GPU acceleration and memory-efficient frame handling.
- Model Input Alignment: Precise reshaping and validation of CoreML MLMultiArray.
- Visual Artifacts: Handled using CoreGraphics blend modes and gradient masks.

## Future Scope
- Live camera input support
- Real-time streaming optimization
- Transformer-based multilingual support
- Enhanced animation blending

##  License
MIT License

---

🔗 Original paper: [Wav2Lip: Accurately Lip-syncing Videos In The Wild](https://arxiv.org/abs/2008.10010)

