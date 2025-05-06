import AVFoundation
import SwiftUI
import CoreML

class AudioManager: NSObject, ObservableObject {
    @Published var currentAmplitude: CGFloat = 0
    private var audioEngine: AVAudioEngine?
    private let audioProcessor = AudioProcessor()
    
    // Audio processing constants
    private let sampleRate: Double = 16000
    private let bufferSize: AVAudioFrameCount = 1024
    private let hopLength: Int = 200  // For 25fps video
    
    // Audio buffer management
    private var audioBuffer: [Float] = []
    private let requiredSamples = 3800  // (16 frames * 200 hop length) + 800 FFT size
    
    // Processing queues
    private let processingQueue = DispatchQueue(label: "com.wav2lip.processing", qos: .userInitiated)
    private let melQueue = DispatchQueue(label: "com.wav2lip.melQueue", qos: .userInitiated)
    
    override init() {
        super.init()
        setupAudioSession()
    }
    
    private func setupAudioSession() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothA2DP])
            try session.setActive(true)
            print("✅ Audio session configured with hardware sample rate: \(session.sampleRate) Hz")
            
            // Check microphone permission
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized:
                print("✅ Microphone access authorized")
            case .denied:
                print("❌ Microphone access denied")
            case .notDetermined:
                print("⚠️ Microphone access not determined")
                requestMicrophoneAccess()
            case .restricted:
                print("❌ Microphone access restricted")
            @unknown default:
                print("❌ Unknown microphone authorization status")
            }
        } catch {
            print("❌ Failed to configure audio session: \(error.localizedDescription)")
        }
    }
    
    private func requestMicrophoneAccess() {
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async {
                if granted {
                    print("✅ Microphone access granted")
                } else {
                    print("❌ Microphone access denied by user")
                }
            }
        }
    }
    
    func startRecording(onMelSpectrogram: @escaping (MLMultiArray) -> Void) {
        print("🎤 Starting audio capture...")
        
        // Reset audio engine if it exists
        if let existingEngine = audioEngine {
            existingEngine.stop()
            existingEngine.inputNode.removeTap(onBus: 0)
        }
        
        audioEngine = AVAudioEngine()
        guard let audioEngine = audioEngine else { return }
        
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)
        
        // Create format converter for 16kHz
        let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                       sampleRate: sampleRate,
                                       channels: 1,
                                       interleaved: false)!
        
        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            print("❌ Failed to create audio converter")
            return
        }
        
        // Install tap with buffer size for ~200ms of audio
        let bufferSize = AVAudioFrameCount(inputFormat.sampleRate * 0.2)
        
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            
            // Convert buffer to 16kHz
            let frameCount = AVAudioFrameCount(Double(buffer.frameLength) * self.sampleRate / inputFormat.sampleRate)
            guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat,
                                                       frameCapacity: frameCount) else {
                return
            }
            
            var error: NSError?
            converter.convert(to: convertedBuffer, error: &error, withInputFrom: { inNumPackets, outStatus in
                outStatus.pointee = .haveData
                return buffer
            })
            
            if let error = error {
                print("❌ Conversion error: \(error.localizedDescription)")
                return
            }
            
            // Process converted audio
            self.processingQueue.async {
                let samples = self.convertBufferToSamples(convertedBuffer)
                
                // Update amplitude visualization
                self.updateAmplitude(from: samples)
                
                // Add to buffer
                self.audioBuffer.append(contentsOf: samples)
                
                // Process when we have enough samples
                if self.audioBuffer.count >= self.requiredSamples {
                    let processingSamples = Array(self.audioBuffer.prefix(self.requiredSamples))
                    self.audioBuffer.removeFirst(samples.count)
                    
                    // Generate mel spectrogram
                    self.melQueue.async {
                        if let melSpec = self.audioProcessor.audioToMelSpectrogram(audioSamples: processingSamples) {
                            DispatchQueue.main.async {
                                onMelSpectrogram(melSpec)
                            }
                        }
                    }
                }
            }
        }
        
        do {
            try audioEngine.start()
            print("✅ Audio engine started successfully")
        } catch {
            print("❌ Failed to start audio engine: \(error.localizedDescription)")
        }
    }
    
    private func convertBufferToSamples(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: channelData,
                                        count: Int(buffer.frameLength)))
    }
    
    private func updateAmplitude(from samples: [Float]) {
        let sum = samples.reduce(0) { $0 + abs($1) }
        let average = sum / Float(samples.count)
        
        DispatchQueue.main.async {
            self.currentAmplitude = CGFloat(average * 50)  // Scale for visualization
        }
    }
    
    func stopRecording() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioBuffer.removeAll()
        print("🛑 Audio recording stopped")
    }
}
