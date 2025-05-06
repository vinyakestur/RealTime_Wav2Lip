import SwiftUI
import AVFoundation
import CoreML
import UIKit
import AVKit

struct ContentView: View {
    @StateObject private var audioManager = AudioManager()
    @StateObject private var audioSpectrogram = AudioSpectrogram()
    @StateObject private var lipSyncProcessor = LipSyncProcessor()
    private let faceProcessor = FaceProcessor()
    @State private var outputImage: UIImage?
    @State private var isProcessing = false
    @State private var originalFaceImage: UIImage?
    
    // Initialize face processing when view appears
    private func initializeFaceProcessing() {
        print("🔍 Attempting to load Elon image...")
        
        guard let originalImage = UIImage(named: "Elon") else {
            print("❌ Failed to load image")
            return
        }
        
        // Process the face data from the static image
        let faceData = faceProcessor.extractFaceSequenceForLipSync(from: originalImage)
        
        if let faceSequence = faceData.sequence {
            print("✅ Face sequence extracted successfully")
            print("Shape: \(faceSequence.shape.map { $0.intValue })")
            
            // Start continuous audio processing with the face sequence and original image
            startContinuousLipSync(with: faceSequence, originalImage: faceData.originalImage)
        } else {
            print("❌ Failed to extract face sequence")
        }
    }
    
    private func verifyModelSetup() {
        print("🔍 Starting Wav2Lip model verification...")
        
        _ = LipSyncProcessor()
        
        do {
            let config = MLModelConfiguration()
            let model = try weight_quantized_wave2lip_model(configuration: config)
            print("✅ Model loaded successfully")
            
            // Print model details
            let description = model.model.modelDescription
            print("📊 Model configuration:")
            print("- Input shapes:")
            description.inputDescriptionsByName.forEach { name, desc in
                print("  • \(name): \(desc.type), shape: \(desc.multiArrayConstraint?.shape ?? [])")
            }
            print("- Output shapes:")
            description.outputDescriptionsByName.forEach { name, desc in
                print("  • \(name): \(desc.type), shape: \(desc.multiArrayConstraint?.shape ?? [])")
            }
            
            // Verify model parameters
            print("📝 Model parameters:")
            print("- Compute units: \(config.computeUnits.rawValue)")
            print("- Allow low precision: \(config.allowLowPrecisionAccumulationOnGPU)")
            
        } catch {
            print("❌ Failed to load model: \(error.localizedDescription)")
        }
    }
    
    private func startContinuousLipSync(with faceSequence: MLMultiArray, originalImage: UIImage) {
        // Start continuous audio recording
        audioManager.startRecording { melSpectrogram in
            print("✅ Mel Spectrogram Generated with shape:", melSpectrogram.shape.map { $0.intValue })
            
            // Update spectrogram visualization
            audioSpectrogram.updateSpectrogram(from: melSpectrogram)
            
            // Process lip sync with latest audio, original face image, and face input
            if let outputImage = lipSyncProcessor.performLipSync(
                audioInput: melSpectrogram,
                originalFaceImage: originalImage,
                faceInput: faceSequence
            ) {
                // Update UI with new frame
                DispatchQueue.main.async {
                    self.lipSyncProcessor.outputImage = outputImage
                }
            }
        }
    }
    
    var body: some View {
        ZStack {
            // Background gradient
            LinearGradient(
                gradient: Gradient(colors: [
                    Color.blue, Color.purple, Color.orange
                ]),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            ScrollView {
                VStack(spacing: 30) {
                    // Title
                    Text("Wav2Lip Animator")
                        .font(.system(size: 32, weight: .bold))
                        .foregroundColor(.white)
                        .padding(.top, 40)
                    
                    // Face Output Container
                    VStack {
                        if let outputImage = lipSyncProcessor.outputImage {
                            Image(uiImage: outputImage)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 300)
                                .clipShape(RoundedRectangle(cornerRadius: 15))
                                .shadow(radius: 10)
                        } else {
                            Image("Elon")
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 300)
                                .clipShape(RoundedRectangle(cornerRadius: 15))
                                .shadow(radius: 10)
                        }
                    }
                    .padding(.horizontal)
                    .background(Color.black.opacity(0.2))
                    .cornerRadius(20)
                    .padding(.horizontal)
                    
                    // Audio Visualization Container
                    VStack(spacing: 20) {
                        // Spectrogram
                        VStack(spacing: 10) {
                            Text("Audio Spectrogram")
                                .font(.headline)
                                .foregroundColor(.white)
                            
                            if let spectrogramImage = audioSpectrogram.outputImage {
                                Image(uiImage: UIImage(cgImage: spectrogramImage))
                                    .resizable()
                                    .scaledToFit()
                                    .frame(height: 150)
                                    .background(Color.black.opacity(0.2))
                                    .clipShape(RoundedRectangle(cornerRadius: 15))
                                    .shadow(radius: 5)
                            }
                        }
                        .padding(.horizontal)
                        
                        // Audio Waveform
                        VStack(spacing: 10) {
                            Text("Audio Input")
                                .font(.headline)
                                .foregroundColor(.white)
                            
                            AudioWaveformView(amplitude: audioManager.currentAmplitude)
                                .frame(height: 60)
                                .padding(.horizontal)
                                .background(
                                    RoundedRectangle(cornerRadius: 15)
                                        .fill(Color.white.opacity(0.2))
                                )
                                .shadow(radius: 5)
                        }
                        .padding(.horizontal)
                    }
                    .padding(.vertical)
                }
                .padding(.bottom, 30)
            }
        }
        .onAppear {
            verifyModelSetup()
            audioSpectrogram.startRunning()
            initializeFaceProcessing()
        }
        .onDisappear {
            audioManager.stopRecording()
            audioSpectrogram.captureSession.stopRunning()
        }
    }
}
