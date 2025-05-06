//
//  AudioSpectrogram.swift
//  Frontend_wav2lip
//
//  Created by Vibha Kestur Tumakuru Arun Kumar on 2/3/25.
//

import SwiftUI
import Accelerate
import AVFoundation
import CoreImage
import UIKit
import CoreML

class AudioSpectrogram: NSObject, ObservableObject {
    
    enum Mode: String, CaseIterable, Identifiable {
        case linear
        case mel
        
        var id: Self { self }
    }
    
    @Published var mode = Mode.linear
    @Published var gain: Double = 0.025
    @Published var zeroReference: Double = 1000
    @Published var outputImage: CGImage?
    
    static let sampleCount = 1024
    static let bufferCount = 768
    static let hopCount = 512

    let captureSession = AVCaptureSession()
    let audioOutput = AVCaptureAudioDataOutput()
    let captureQueue = DispatchQueue(label: "captureQueue", qos: .userInitiated)
    let sessionQueue = DispatchQueue(label: "sessionQueue")
    
    let forwardDCT = vDSP.DCT(count: sampleCount, transformType: .II)!
    let hanningWindow = vDSP.window(ofType: Float.self, usingSequence: .hanningDenormalized, count: sampleCount, isHalfWindow: false)
    
    var nyquistFrequency: Float?
    var rawAudioData = [Int16]()
    var frequencyDomainValues = [Float](repeating: 0, count: bufferCount * sampleCount)
    
    let redBuffer = vImage.PixelBuffer<vImage.PlanarF>(width: sampleCount, height: bufferCount)
    let greenBuffer = vImage.PixelBuffer<vImage.PlanarF>(width: sampleCount, height: bufferCount)
    let blueBuffer = vImage.PixelBuffer<vImage.PlanarF>(width: sampleCount, height: bufferCount)
    let rgbImageBuffer = vImage.PixelBuffer<vImage.InterleavedFx3>(width: sampleCount, height: bufferCount)

    var timeDomainBuffer = [Float](repeating: 0, count: sampleCount)
    var frequencyDomainBuffer = [Float](repeating: 0, count: sampleCount)
    
    let rgbImageFormat = vImage_CGImageFormat(
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        colorSpace: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
    )!
    
    override init() {
        super.init()
        configureCaptureSession()
        audioOutput.setSampleBufferDelegate(self, queue: captureQueue)
    }
    
    func processData(values: [Int16]) {
        vDSP.convertElements(of: values, to: &timeDomainBuffer)
        vDSP.multiply(timeDomainBuffer, hanningWindow, result: &timeDomainBuffer)
        forwardDCT.transform(timeDomainBuffer, result: &frequencyDomainBuffer)
        vDSP.absolute(frequencyDomainBuffer, result: &frequencyDomainBuffer)
        
        switch mode {
        case .linear:
            vDSP.convert(amplitude: frequencyDomainBuffer, toDecibels: &frequencyDomainBuffer, zeroReference: Float(zeroReference))
        case .mel:
            // Implement mel spectrogram processing
            break
        }

        vDSP.multiply(Float(gain), frequencyDomainBuffer, result: &frequencyDomainBuffer)
        
        if frequencyDomainValues.count > AudioSpectrogram.sampleCount {
            frequencyDomainValues.removeFirst(AudioSpectrogram.sampleCount)
        }
        
        frequencyDomainValues.append(contentsOf: frequencyDomainBuffer)
    }
    
    func makeAudioSpectrogramImage() -> CGImage {
        frequencyDomainValues.withUnsafeMutableBufferPointer {
            let planarImageBuffer = vImage.PixelBuffer(
                data: $0.baseAddress!,
                width: AudioSpectrogram.sampleCount,
                height: AudioSpectrogram.bufferCount,
                byteCountPerRow: AudioSpectrogram.sampleCount * MemoryLayout<Float>.stride,
                pixelFormat: vImage.PlanarF.self)
            
            AudioSpectrogram.multidimensionalLookupTable.apply(
                sources: [planarImageBuffer],
                destinations: [redBuffer, greenBuffer, blueBuffer],
                interpolation: .half)
            
            rgbImageBuffer.interleave(
                planarSourceBuffers: [redBuffer, greenBuffer, blueBuffer])
        }
        
        return rgbImageBuffer.makeCGImage(cgImageFormat: rgbImageFormat) ?? AudioSpectrogram.emptyCGImage
    }
    
    func configureCaptureSession() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            break
        case .notDetermined:
            sessionQueue.suspend()
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                if !granted {
                    fatalError("App requires microphone access.")
                } else {
                    self.configureCaptureSession()
                    self.sessionQueue.resume()
                }
            }
            return
        default:
            fatalError("App requires microphone access.")
        }
        
        captureSession.beginConfiguration()
        
        #if os(macOS)
        audioOutput.audioSettings = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMBitDepthKey: 16,
            AVNumberOfChannelsKey: 1
        ]
        #endif
        
        if captureSession.canAddOutput(audioOutput) {
            captureSession.addOutput(audioOutput)
        } else {
            fatalError("Can't add audioOutput.")
        }
        
        guard let microphone = AVCaptureDevice.default(.microphone, for: .audio,
                                                      position: .unspecified),
              let microphoneInput = try? AVCaptureDeviceInput(device: microphone) else {
            fatalError("Can't create microphone.")
        }
        
        if captureSession.canAddInput(microphoneInput) {
            captureSession.addInput(microphoneInput)
        }
        
        captureSession.commitConfiguration()
    }
    
    func startRunning() {
        sessionQueue.async {
            if AVCaptureDevice.authorizationStatus(for: .audio) == .authorized {
                self.captureSession.startRunning()
            }
        }
    }
    
    static var multidimensionalLookupTable: vImage.MultidimensionalLookupTable = {
        let entriesPerChannel: Int = 32
        let srcChannelCount = 1
        let destChannelCount: Int = 3
        
        let tableData = [UInt16](unsafeUninitializedCapacity: entriesPerChannel * destChannelCount) { buffer, count in
            let multiplier = CGFloat(UInt16.max)
            var bufferIndex = 0
            
            for gray in 0..<entriesPerChannel {
                let normalizedValue = CGFloat(gray) / CGFloat(entriesPerChannel - 1)
                let hue = 0.6666 - (0.6666 * normalizedValue)
                let brightness = sqrt(normalizedValue)
                
                let color = UIColor(hue: hue,
                                  saturation: 1,
                                  brightness: brightness,
                                  alpha: 1)
                
                var red: CGFloat = 0
                var green: CGFloat = 0
                var blue: CGFloat = 0
                var alpha: CGFloat = 0
                
                color.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
                
                buffer[bufferIndex] = UInt16(green * multiplier)
                bufferIndex += 1
                buffer[bufferIndex] = UInt16(red * multiplier)
                bufferIndex += 1
                buffer[bufferIndex] = UInt16(blue * multiplier)
                bufferIndex += 1
            }
            count = entriesPerChannel * destChannelCount
        }
        
        return vImage.MultidimensionalLookupTable(
            entryCountPerSourceChannel: [UInt8(entriesPerChannel)],
            destinationChannelCount: destChannelCount,
            data: tableData
        )
    }()
    
    static var emptyCGImage: CGImage = {
        let width = 1
        let height = 1
        let bitsPerComponent = 8
        let bytesPerRow = 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ),
        let image = context.makeImage() else {
            fatalError("Could not create empty CGImage")
        }
        
        return image
    }()
    
    // Update spectrogram with new mel data
    func updateSpectrogram(from melSpectrogram: MLMultiArray) {
        DispatchQueue.main.async {
            self.outputImage = self.generateSpectrogram(from: melSpectrogram)
        }
    }
    
    private func generateSpectrogram(from melSpectrogram: MLMultiArray) -> CGImage? {
        // Convert MLMultiArray to image
        let width = 80  // Mel bins
        let height = 16 // Time frames
        
        // Create pixel data buffer
        var pixelData = [UInt8](repeating: 0, count: width * height * 4) // RGBA
        
        // Extract values from melSpectrogram (using first batch)
        for timeIdx in 0..<height {
            for melBin in 0..<width {
                let indices = [0, 0, melBin, timeIdx] as [NSNumber]
                let value = melSpectrogram[indices].floatValue
                
                // Convert to grayscale RGBA
                let pixelValue = UInt8(min(max(value * 255.0, 0), 255))
                let pixelIndex = (timeIdx * width + melBin) * 4
                
                // Set RGBA values (grayscale)
                pixelData[pixelIndex] = pixelValue     // R
                pixelData[pixelIndex + 1] = pixelValue // G
                pixelData[pixelIndex + 2] = pixelValue // B
                pixelData[pixelIndex + 3] = 255        // A
            }
        }
        
        // Create CGImage
        let bitsPerComponent = 8
        let bitsPerPixel = 32
        let bytesPerRow = width * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let provider = CGDataProvider(data: Data(pixelData) as CFData),
              let cgImage = CGImage(width: width,
                                  height: height,
                                  bitsPerComponent: bitsPerComponent,
                                  bitsPerPixel: bitsPerPixel,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: bitmapInfo,
                                  provider: provider,
                                  decode: nil,
                                  shouldInterpolate: false,
                                  intent: .defaultIntent) else {
            print("❌ Failed to create spectrogram image")
            return nil
        }
        
        return cgImage
    }
    
    // Apply colormap to make visualization more appealing
    private func applyColormap(_ value: Float) -> (r: UInt8, g: UInt8, b: UInt8) {
        let v = min(max(value, 0), 1)
        
        // Simple blue-to-red colormap
        return (
            r: UInt8(v * 255),
            g: UInt8((1 - v) * 255),
            b: UInt8(255 - v * 255)
        )
    }
}

extension AudioSpectrogram: AVCaptureAudioDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        var audioBufferList = AudioBufferList()
        var blockBuffer: CMBlockBuffer?

        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: &audioBufferList,
            bufferListSize: MemoryLayout.stride(ofValue: audioBufferList),
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &blockBuffer
        )
        
        guard let data = audioBufferList.mBuffers.mData else { return }
        
        if nyquistFrequency == nil {
            let duration = Float(CMSampleBufferGetDuration(sampleBuffer).value)
            let timescale = Float(CMSampleBufferGetDuration(sampleBuffer).timescale)
            let numsamples = Float(CMSampleBufferGetNumSamples(sampleBuffer))
            nyquistFrequency = 0.5 / (duration / timescale / numsamples)
        }
        
        let actualSampleCount = CMSampleBufferGetNumSamples(sampleBuffer)
        let pointer = data.bindMemory(to: Int16.self, capacity: actualSampleCount)
        let buffer = UnsafeBufferPointer(start: pointer, count: actualSampleCount)
        
        rawAudioData.append(contentsOf: Array(buffer))
        
        while rawAudioData.count >= AudioSpectrogram.sampleCount {
            let dataToProcess = Array(rawAudioData[0..<AudioSpectrogram.sampleCount])
            rawAudioData.removeFirst(AudioSpectrogram.hopCount)
            processData(values: dataToProcess)
        }
        
        DispatchQueue.main.async { [self] in
            outputImage = makeAudioSpectrogramImage()
        }
    }
}
