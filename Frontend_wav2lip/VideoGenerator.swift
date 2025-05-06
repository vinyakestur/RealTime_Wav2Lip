import AVFoundation
import UIKit
import SwiftUI

class VideoGenerator: ObservableObject {
    @Published var currentFrame: UIImage?
    @Published var progress: Float = 0.0
    @Published var isExporting = false
    private var displayLink: CADisplayLink?
    private var frameQueue: [UIImage] = []
    private let maxQueueSize = 5  // Buffer a few frames for smoother playback
    
    // Frame rate management
    private let targetFPS: Double = 25.0  // Match Wav2Lip's output rate
    private var lastFrameTime: CFTimeInterval = 0
    private let frameDuration: CFTimeInterval
    
    init() {
        self.frameDuration = 1.0 / targetFPS
        setupDisplayLink()
    }
    
    private func setupDisplayLink() {
        displayLink = CADisplayLink(target: self, selector: #selector(displayNextFrame))
        displayLink?.preferredFrameRateRange = CAFrameRateRange(minimum: Float(targetFPS),
                                                               maximum: Float(targetFPS),
                                                               preferred: Float(targetFPS))
        displayLink?.add(to: .main, forMode: .common)
        displayLink?.isPaused = true
    }
    
    func start() {
        print("▶️ Starting frame display")
        displayLink?.isPaused = false
    }
    
    func stop() {
        print("⏹️ Stopping frame display")
        displayLink?.isPaused = true
        frameQueue.removeAll()
        currentFrame = nil
    }
    
    func queueFrame(_ frame: UIImage) {
        if frameQueue.count >= maxQueueSize {
            frameQueue.removeFirst()
        }
        
        if frameQueue.count > 1 {
            let lastFrame = frameQueue[frameQueue.count - 2]
            if lastFrame.pngData() == frame.pngData() {
                print("⚠️ Possible duplicate frame detected, allowing minor movement...")
            }
        }


        print("📽️ Queuing new frame for display.")
        frameQueue.append(frame)

        if displayLink?.isPaused == true && !frameQueue.isEmpty {
            start()
        }
    }

    
    @objc private func displayNextFrame(link: CADisplayLink) {
        let currentTime = link.timestamp
        
        // Check if it's time for a new frame
        guard currentTime - lastFrameTime >= frameDuration else { return }
        
        // Display next frame if available
        if !frameQueue.isEmpty {
            DispatchQueue.main.async { [weak self] in
                self?.currentFrame = self?.frameQueue.removeFirst()
            }
            lastFrameTime = currentTime
        } else {
            // Pause display if no frames available
            displayLink?.isPaused = true
        }
    }
    
    // Helper method to create video from frames if needed
    func saveFramesAsVideo(frames: [UIImage], fps: Float = 25.0, completion: @escaping (URL?) -> Void) {
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("output.mov")
        
        // Remove existing file
        try? FileManager.default.removeItem(at: outputURL)
        
        guard let videoWriter = try? AVAssetWriter(outputURL: outputURL, fileType: .mov) else {
            print("❌ Failed to create video writer")
            completion(nil)
            return
        }
        
        let width = Int(frames[0].size.width)
        let height = Int(frames[0].size.height)
        
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]
        
        let videoWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoWriterInput,
                                                          sourcePixelBufferAttributes: nil)
        
        videoWriter.add(videoWriterInput)
        videoWriter.startWriting()
        videoWriter.startSession(atSourceTime: .zero)
        
        let frameDuration = CMTimeMake(value: 1, timescale: CMTimeScale(fps))
        var frameCount = 0
        
        // Process frames in background
        DispatchQueue.global(qos: .userInitiated).async {
            for image in frames {
                if let buffer = image.toPixelBuffer(width: width, height: height) {
                    let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(frameCount))
                    adaptor.append(buffer, withPresentationTime: presentationTime)
                    frameCount += 1
                }
            }
            
            videoWriterInput.markAsFinished()
            videoWriter.finishWriting {
                DispatchQueue.main.async {
                    completion(outputURL)
                }
            }
        }
    }
    
    func createVideo(from frames: [UIImage], fps: Int32 = 25) -> URL? {
        guard !frames.isEmpty else {
            print("❌ No frames to generate video")
            return nil
        }
        
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("lip_sync_output")
            .appendingPathExtension("mp4")
        
        // Remove existing file if any
        try? FileManager.default.removeItem(at: outputURL)
        
        let size = frames[0].size
        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            print("❌ Failed to create asset writer")
            return nil
        }
        
        // Configure video settings
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: size.width,
            AVVideoHeightKey: size.height
        ]
        
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: nil
        )
        
        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        
        // Process frames
        let frameDuration = CMTime(value: 1, timescale: fps)
        var frameCount: Int64 = 0
        
        let queue = DispatchQueue(label: "videoQueue")
        writerInput.requestMediaDataWhenReady(on: queue) {
            for (index, frame) in frames.enumerated() {
                if let buffer = frame.toPixelBuffer() {
                    let presentationTime = CMTime(value: frameCount, timescale: fps)
                    adaptor.append(buffer, withPresentationTime: presentationTime)
                    frameCount += 1
                    
                    DispatchQueue.main.async {
                        self.progress = Float(index) / Float(frames.count)
                    }
                }
            }
            
            writerInput.markAsFinished()
            writer.finishWriting {
                DispatchQueue.main.async {
                    self.progress = 1.0
                    self.isExporting = false
                }
            }
        }
        
        return outputURL
    }
    
    func mergeAudioVideo(videoURL: URL, audioURL: URL) -> URL? {
        let composition = AVMutableComposition()
        
        // Create output URL
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("final_output")
            .appendingPathExtension("mp4")
        
        // Remove existing file if any
        try? FileManager.default.removeItem(at: outputURL)
        
        // Load assets
        let videoAsset = AVURLAsset(url: videoURL)
        let audioAsset = AVURLAsset(url: audioURL)
        
        // Get tracks
        guard let videoTrack = try? composition.addMutableTrack(
            withMediaType: .video,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ),
        let audioTrack = try? composition.addMutableTrack(
            withMediaType: .audio,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ) else {
            print("❌ Failed to create composition tracks")
            return nil
        }
        
        // Insert tracks
        do {
            try videoTrack.insertTimeRange(
                CMTimeRange(start: .zero, duration: videoAsset.duration),
                of: videoAsset.tracks(withMediaType: .video)[0],
                at: .zero
            )
            
            try audioTrack.insertTimeRange(
                CMTimeRange(start: .zero, duration: audioAsset.duration),
                of: audioAsset.tracks(withMediaType: .audio)[0],
                at: .zero
            )
        } catch {
            print("❌ Failed to insert tracks: \(error)")
            return nil
        }
        
        // Export final video
        guard let exporter = AVAssetExportSession(
            asset: composition,
            presetName: AVAssetExportPresetHighestQuality
        ) else {
            print("❌ Failed to create export session")
            return nil
        }
        
        exporter.outputURL = outputURL
        exporter.outputFileType = .mp4
        
        let semaphore = DispatchSemaphore(value: 0)
        exporter.exportAsynchronously {
            semaphore.signal()
        }
        semaphore.wait()
        
        return outputURL
    }
}

// SwiftUI wrapper for displaying frames
struct VideoDisplayView: View {
    @ObservedObject var videoGenerator: VideoGenerator
    
    var body: some View {
        if let frame = videoGenerator.currentFrame {
            Image(uiImage: frame)
                .resizable()
                .aspectRatio(contentMode: .fit)
        } else {
            Color.black
                .aspectRatio(1, contentMode: .fit)
        }
    }
}

extension UIImage {
    func toPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                    kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                       width,
                                       height,
                                       kCVPixelFormatType_32ARGB,
                                       attrs,
                                       &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                              width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
} 
