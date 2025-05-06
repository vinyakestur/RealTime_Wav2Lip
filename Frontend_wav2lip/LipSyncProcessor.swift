import CoreML
import UIKit
import Foundation
import Vision
import Accelerate

class LipSyncProcessor: ObservableObject {
    @Published private(set) var isProcessing = false
    @Published private(set) var error: String?
    @Published var outputImage: UIImage?  // Add published output image
    
    private var model: weight_quantized_wave2lip_model?
    private var staticFaceSequence: MLMultiArray?  // Store face sequence
    private let faceProcessor = FaceProcessor()
    
    @Published private(set) var originalFaceImage: UIImage?
    @Published private(set) var lipRegion: CGRect?
    
    private var generatedFrames: [UIImage] = []
    private let videoGenerator = VideoGenerator()
    
    // Cache for prediction conversion
    private var cachedPredictionBuffer: [UInt8]?
    private var cachedPredictionSize: CGSize?
    
    // Cache for lip merging
    private var cachedLipBuffer: [UInt8]?
    private var cachedLipSize: CGSize?
    private var cachedGradient: CGGradient?
    
    init() {
        setupModel()
        
        // Load and prepare the static face
        if let image = UIImage(named: "Elon") {
            // Store the original image
            originalFaceImage = image
            
            // Extract face data for the model
            let faceData = faceProcessor.extractFaceSequenceForLipSync(from: image)
            staticFaceSequence = faceData.sequence
            originalFaceImage = faceData.originalImage
            lipRegion = faceData.lipRegion
            
            print("✅ Face prepared for lip sync")
            if let sequence = staticFaceSequence {
                print("📊 Face sequence shape: \(sequence.shape.map { $0.intValue })")
            }
        } else {
            print("❌ Failed to load Elon image")
        }
    }
    
    private func setupModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Prefer full GPU
            config.allowLowPrecisionAccumulationOnGPU = true  // Enable low-precision GPU acceleration

            model = try weight_quantized_wave2lip_model(configuration: config)
            print("✅ Wav2Lip model loaded successfully on GPU")

            // Debugging: Log Compute Unit being used
            print("📝 Model running on:", config.computeUnits.rawValue)

        } catch {
            print("❌ Failed to load Wav2Lip model: \(error.localizedDescription)")
        }
    }
    
    private func areImagesSimilar(_ img1: UIImage, _ img2: UIImage) -> Bool {
        guard let data1 = img1.pngData(), let data2 = img2.pngData() else {
            return false
        }
        return data1 == data2  // True if identical
    }
    
    // Cache face detection result and original image data
    private var cachedFaceRect: CGRect?
    private var cachedOriginalCGImage: CGImage?
    private var cachedOriginalBuffer: [UInt8]?
    private var cachedOriginalSize: CGSize?
    
    private func mergePredictedFaceOntoOriginal(_ predictedFace: UIImage, originalImage: UIImage) -> UIImage? {
        let startTime = Date()
        print("🔄 Starting optimized face merging process...")
        
        // Step 1: Face Detection (Cached)
        let detectionStartTime = Date()
        if cachedFaceRect == nil {
            guard let faceObservation = faceProcessor.detectFaceInternal(in: originalImage) else {
                print("❌ No face detected in original image")
                return nil
            }
            
            let imageSize = originalImage.size
            let faceBox = faceObservation.boundingBox
            let faceX = faceBox.origin.x * imageSize.width
            let faceY = (1 - faceBox.origin.y - faceBox.height) * imageSize.height
            let faceWidth = faceBox.width * imageSize.width
            let faceHeight = faceBox.height * imageSize.height
            
            cachedFaceRect = CGRect(
                x: faceX,
                y: faceY,
                width: faceWidth,
                height: faceHeight
            )
        }
        let detectionTime = Date().timeIntervalSince(detectionStartTime) * 1000
        print("⏱ Face detection time (cached): \(detectionTime)ms")
        
        guard let faceRect = cachedFaceRect else { return nil }
        
        // Step 2: Get Original Image Data (Cached)
        let bufferStartTime = Date()
        if cachedOriginalBuffer == nil || cachedOriginalSize != originalImage.size {
            guard let originalCGImage = originalImage.cgImage else {
                print("❌ Failed to get original CGImage")
                return nil
            }
            
            let width = Int(originalImage.size.width)
            let height = Int(originalImage.size.height)
            let bytesPerRow = width * 4
            
            var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
            
            guard let context = CGContext(data: &pixelData,
                                        width: width,
                                        height: height,
                                        bitsPerComponent: 8,
                                        bytesPerRow: bytesPerRow,
                                        space: colorSpace,
                                        bitmapInfo: bitmapInfo.rawValue) else {
                print("❌ Failed to create context")
                return nil
            }
            
            context.draw(originalCGImage, in: CGRect(origin: .zero, size: originalImage.size))
            cachedOriginalBuffer = pixelData
            cachedOriginalSize = originalImage.size
            cachedOriginalCGImage = originalCGImage
        }
        let bufferTime = Date().timeIntervalSince(bufferStartTime) * 1000
        print("⏱ Original buffer preparation time (cached): \(bufferTime)ms")
        
        // Step 3: Get Predicted Face Data
        let predStartTime = Date()
        guard let predictedCGImage = predictedFace.cgImage else {
            print("❌ Failed to get predicted CGImage")
            return nil
        }
        
        let faceWidthInt = Int(faceRect.width)
        let faceHeightInt = Int(faceRect.height)
        let bytesPerRow = faceWidthInt * 4
        var predictedData = [UInt8](repeating: 0, count: faceHeightInt * bytesPerRow)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        
        guard let predictedContext = CGContext(data: &predictedData,
                                             width: faceWidthInt,
                                             height: faceHeightInt,
                                             bitsPerComponent: 8,
                                             bytesPerRow: bytesPerRow,
                                             space: colorSpace,
                                             bitmapInfo: bitmapInfo.rawValue) else {
            print("❌ Failed to create predicted face context")
            return nil
        }
        
        predictedContext.draw(predictedCGImage, in: CGRect(origin: .zero, size: CGSize(width: faceWidthInt, height: faceHeightInt)))
        let predTime = Date().timeIntervalSince(predStartTime) * 1000
        print("⏱ Predicted face preparation time: \(predTime)ms")
        
        // Step 4: Fast Pixel Copying using SIMD
        let copyStartTime = Date()
        guard var originalBuffer = cachedOriginalBuffer else { return nil }
        let width = Int(originalImage.size.width)
        let faceXInt = Int(faceRect.minX)
        let faceYInt = Int(faceRect.minY)
        
        // Use SIMD for faster pixel copying
        let faceWidthBytes = faceWidthInt * 4
        for y in 0..<faceHeightInt {
            let originalOffset = ((faceYInt + y) * width + faceXInt) * 4
            let predictedOffset = y * faceWidthBytes
            
            // Copy entire row at once using UnsafeMutableRawPointer
            predictedData.withUnsafeBufferPointer { predictedPtr in
                originalBuffer.withUnsafeMutableBufferPointer { originalPtr in
                    let predictedBase = predictedPtr.baseAddress! + predictedOffset
                    let originalBase = originalPtr.baseAddress! + originalOffset
                    memcpy(originalBase, predictedBase, faceWidthBytes)
                }
            }
        }
        let copyTime = Date().timeIntervalSince(copyStartTime) * 1000
        print("⏱ Pixel copying time (SIMD): \(copyTime)ms")
        
        // Step 5: Create Final Image
        let finalStartTime = Date()
        guard let provider = CGDataProvider(data: Data(originalBuffer) as CFData),
              let cgImage = CGImage(width: width,
                                  height: Int(originalImage.size.height),
                                  bitsPerComponent: 8,
                                  bitsPerPixel: 32,
                                  bytesPerRow: width * 4,
                                  space: colorSpace,
                                  bitmapInfo: bitmapInfo,
                                  provider: provider,
                                  decode: nil,
                                  shouldInterpolate: false,
                                  intent: .defaultIntent) else {
            print("❌ Failed to create final image")
            return nil
        }
        let finalTime = Date().timeIntervalSince(finalStartTime) * 1000
        print("⏱ Final image creation time: \(finalTime)ms")
        
        let totalTime = Date().timeIntervalSince(startTime) * 1000
        print("\n📊 Optimized Face Merging Performance Summary:")
        print("Total time: \(totalTime)ms")
        print("\nTime breakdown:")
        print("- Face detection (cached): \(detectionTime)ms (\(detectionTime/totalTime*100)%)")
        print("- Original buffer (cached): \(bufferTime)ms (\(bufferTime/totalTime*100)%)")
        print("- Predicted face preparation: \(predTime)ms (\(predTime/totalTime*100)%)")
        print("- Pixel copying (SIMD): \(copyTime)ms (\(copyTime/totalTime*100)%)")
        print("- Final image creation: \(finalTime)ms (\(finalTime/totalTime*100)%)")
        
        return UIImage(cgImage: cgImage)
    }

    func performLipSync(audioInput: MLMultiArray, originalFaceImage: UIImage, faceInput: MLMultiArray) -> UIImage? {
        let totalStartTime = Date()
        print("\n🔄 Starting lip sync processing pipeline...")
        
        guard !isProcessing else {
            print("⚠️ Already processing previous frame")
            return nil
        }

        guard let model = model else {
            print("❌ Model not loaded")
            return nil
        }

        if staticFaceSequence == nil {
            staticFaceSequence = faceInput
            print("✅ Face sequence stored with shape: \(faceInput.shape.map { $0.intValue })")
        }

        guard let finalFaceInput = staticFaceSequence else {
            print("❌ No face sequence available for processing")
            return nil
        }

        isProcessing = true
        error = nil

        do {
            // Step 1: Input Preparation
            let prepStartTime = Date()
            let modelInput = weight_quantized_wave2lip_modelInput(
                audio_sequences: audioInput,
                face_sequences: finalFaceInput
            )
            let prepTime = Date().timeIntervalSince(prepStartTime) * 1000
            print("⏱ Input preparation time: \(prepTime)ms")

            // Step 2: Model Inference
            let inferenceStartTime = Date()
            let prediction = try model.prediction(input: modelInput)
            let predictedLips = prediction.var_855
            let inferenceTime = Date().timeIntervalSince(inferenceStartTime) * 1000
            print("⏱ Model inference time: \(inferenceTime)ms")

            // Step 3: Convert prediction to image
            let conversionStartTime = Date()
            guard let faceImage = convertPredictionToImage(predictedLips) else {
                print("❌ Failed to convert prediction to image")
                isProcessing = false
                return nil
            }
            let conversionTime = Date().timeIntervalSince(conversionStartTime) * 1000
            print("⏱ Prediction to image conversion time: \(conversionTime)ms")

            // Step 4: Merge predicted face onto original
            let mergeStartTime = Date()
            guard let mergedImage = mergePredictedFaceOntoOriginal(faceImage, originalImage: originalFaceImage) else {
                print("❌ Failed to blend predicted lips onto the original face")
                isProcessing = false
                return nil
            }
            let mergeTime = Date().timeIntervalSince(mergeStartTime) * 1000
            print("⏱ Face merging time: \(mergeTime)ms")

            // Step 5: UI Update
            let uiStartTime = Date()
            DispatchQueue.main.async {
                self.outputImage = mergedImage
            }
            let uiTime = Date().timeIntervalSince(uiStartTime) * 1000
            print("⏱ UI update time: \(uiTime)ms")

            isProcessing = false

            // Total processing time
            let totalTime = Date().timeIntervalSince(totalStartTime) * 1000
            print("\n📊 Performance Summary:")
            print("Total processing time: \(totalTime)ms")
            print("Current FPS: \(1000/totalTime)")
            print("Target FPS: 25")
            print("\nTime breakdown:")
            print("- Input preparation: \(prepTime)ms (\(prepTime/totalTime*100)%)")
            print("- Model inference: \(inferenceTime)ms (\(inferenceTime/totalTime*100)%)")
            print("- Image conversion: \(conversionTime)ms (\(conversionTime/totalTime*100)%)")
            print("- Face merging: \(mergeTime)ms (\(mergeTime/totalTime*100)%)")
            print("- UI update: \(uiTime)ms (\(uiTime/totalTime*100)%)")

            return mergedImage

        } catch {
            print("❌ Frame processing failed: \(error.localizedDescription)")
            self.error = error.localizedDescription
            isProcessing = false
            return nil
        }
    }


    func prepareStaticFace(_ image: UIImage) -> Bool {
        print("🔍 Processing static face image...")
        
        // Store the original image
        originalFaceImage = image
        
        // Extract face data for the model
        let faceData = faceProcessor.extractFaceSequenceForLipSync(from: image)
        staticFaceSequence = faceData.sequence
        originalFaceImage = faceData.originalImage
        lipRegion = faceData.lipRegion
        
        guard staticFaceSequence != nil else {
            print("❌ Face processing failed")
            return false
        }
        
        print("✅ Static face prepared successfully")
        print("📊 Face sequence shape: \(staticFaceSequence!.shape.map { $0.intValue })")
        return true
    }
    
    
    private func convertMLArrayToImage(_ array: MLMultiArray) -> UIImage? {
        let width = 96
        let height = 96
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        
        // Get first frame's RGB channels
        for y in 0..<height {
            for x in 0..<width {
                let r = array[[0, 0, y, x] as [NSNumber]].floatValue
                let g = array[[0, 1, y, x] as [NSNumber]].floatValue
                let b = array[[0, 2, y, x] as [NSNumber]].floatValue
                
                let pixelIndex = (y * width + x) * 4
                pixelData[pixelIndex] = UInt8(max(0, min(255, r * 255.0)))
                pixelData[pixelIndex + 1] = UInt8(max(0, min(255, g * 255.0)))
                pixelData[pixelIndex + 2] = UInt8(max(0, min(255, b * 255.0)))
                pixelData[pixelIndex + 3] = 255
            }
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        
        guard let provider = CGDataProvider(data: Data(pixelData) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            print("❌ Failed to create image from array")
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
    

    
    private func convertPredictionToImage(_ outputArray: MLMultiArray) -> UIImage? {
        let startTime = Date()
        print("🎨 Converting prediction to image...")

        let width = 96
        let height = 96
        let channels = 3 // RGB
        
        // Validate shape once
        guard outputArray.shape.count == 4,
              outputArray.shape[1].intValue == channels,
              outputArray.shape[2].intValue == height,
              outputArray.shape[3].intValue == width else {
            print("❌ Error: Unexpected output array shape:", outputArray.shape.map { $0.intValue })
            return nil
        }

        // Reuse buffer if possible
        let bufferStartTime = Date()
        let bufferSize = width * height * 4
        if cachedPredictionBuffer == nil || cachedPredictionSize != CGSize(width: width, height: height) {
            cachedPredictionBuffer = [UInt8](repeating: 0, count: bufferSize)
            cachedPredictionSize = CGSize(width: width, height: height)
        }
        let bufferTime = Date().timeIntervalSince(bufferStartTime) * 1000
        print("⏱ Buffer preparation time: \(bufferTime)ms")
        
        guard var pixelData = cachedPredictionBuffer else { return nil }
        
        // Get bulk access to MLMultiArray data
        let rowSize = width * 4
        
        // Pre-allocate arrays for all rows to avoid repeated allocations
        let arrayStartTime = Date()
        var rInput = [Float](repeating: 0, count: width * height)
        var gInput = [Float](repeating: 0, count: width * height)
        var bInput = [Float](repeating: 0, count: width * height)
        let arrayTime = Date().timeIntervalSince(arrayStartTime) * 1000
        print("⏱ Array allocation time: \(arrayTime)ms")
        
        // Extract all RGB values at once to minimize MLMultiArray access overhead
        let extractStartTime = Date()
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                rInput[idx] = outputArray[[0, 0, y, x] as [NSNumber]].floatValue
                gInput[idx] = outputArray[[0, 1, y, x] as [NSNumber]].floatValue
                bInput[idx] = outputArray[[0, 2, y, x] as [NSNumber]].floatValue
            }
        }
        let extractTime = Date().timeIntervalSince(extractStartTime) * 1000
        print("⏱ MLMultiArray extraction time: \(extractTime)ms")
        
        // Process pixels using SIMD with minimal allocations
        let processStartTime = Date()
        pixelData.withUnsafeMutableBufferPointer { ptr in
            var scale: Float = 255.0
            
            // Create separate output arrays for SIMD operations
            var rOutput = [Float](repeating: 0, count: width * height)
            var gOutput = [Float](repeating: 0, count: width * height)
            var bOutput = [Float](repeating: 0, count: width * height)
            
            // Process all pixels at once using SIMD
            vDSP_vsmul(&rInput, 1, &scale, &rOutput, 1, vDSP_Length(width * height))
            vDSP_vsmul(&gInput, 1, &scale, &gOutput, 1, vDSP_Length(width * height))
            vDSP_vsmul(&bInput, 1, &scale, &bOutput, 1, vDSP_Length(width * height))
            
            // Copy to output buffer with minimal bounds checking
            for i in 0..<(width * height) {
                let y = i / width
                let x = i % width
                let pixelOffset = (y * width + x) * 4
                
                ptr[pixelOffset] = UInt8(max(0, min(255, bOutput[i])))     // B
                ptr[pixelOffset + 1] = UInt8(max(0, min(255, gOutput[i]))) // G
                ptr[pixelOffset + 2] = UInt8(max(0, min(255, rOutput[i]))) // R
                ptr[pixelOffset + 3] = 255                                // Alpha
            }
        }
        let processTime = Date().timeIntervalSince(processStartTime) * 1000
        print("⏱ Pixel processing time: \(processTime)ms")

        // Create image with optimized settings
        let imageStartTime = Date()
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)

        guard let provider = CGDataProvider(data: Data(pixelData) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: rowSize,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: false, // Disable interpolation for speed
                intent: .defaultIntent
              ) else {
            print("❌ Failed to create CGImage from pixel data")
            return nil
        }
        let imageTime = Date().timeIntervalSince(imageStartTime) * 1000
        print("⏱ Image creation time: \(imageTime)ms")

        let processingTime = Date().timeIntervalSince(startTime) * 1000
        print("⏱ Total prediction conversion time: \(processingTime)ms")
        
        return UIImage(cgImage: cgImage)
    }

    private func mergePredictedLips(_ predictedLips: UIImage, withOriginal originalFace: MLMultiArray) -> UIImage? {
        let startTime = Date()
        print("🔄 Starting optimized lip merging...")
        
        let targetSize = CGSize(width: 96, height: 96)
        
        // Step 1: Convert original face (cached)
        let originalStartTime = Date()
        guard let originalImage = convertMLArrayToImage(originalFace) else {
            print("❌ Failed to convert original face back to image")
            return nil
        }
        let originalTime = Date().timeIntervalSince(originalStartTime) * 1000
        print("⏱ Original face conversion time: \(originalTime)ms")
        
        // Step 2: Prepare buffers
        let bufferStartTime = Date()
        let bufferSize = Int(targetSize.width * targetSize.height * 4)
        if cachedLipBuffer == nil || cachedLipSize != targetSize {
            cachedLipBuffer = [UInt8](repeating: 0, count: bufferSize)
            cachedLipSize = targetSize
        }
        
        guard var pixelData = cachedLipBuffer else { return nil }
        
        // Step 3: Draw original face to buffer
        guard let originalCGImage = originalImage.cgImage else { return nil }
        let context = CGContext(data: &pixelData,
                              width: Int(targetSize.width),
                              height: Int(targetSize.height),
                              bitsPerComponent: 8,
                              bytesPerRow: Int(targetSize.width * 4),
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        
        context?.draw(originalCGImage, in: CGRect(origin: .zero, size: targetSize))
        let bufferTime = Date().timeIntervalSince(bufferStartTime) * 1000
        print("⏱ Buffer preparation time: \(bufferTime)ms")
        
        // Step 4: Define lip region
        let lipYStart = targetSize.height * 0.58
        let lipHeight = targetSize.height * 0.25
        let lipXStart = targetSize.width * 0.25
        let lipWidth = targetSize.width * 0.50
        let lipRect = CGRect(x: lipXStart, y: lipYStart, width: lipWidth, height: lipHeight)
        
        // Step 5: Create gradient mask (cached)
        let gradientStartTime = Date()
        if cachedGradient == nil {
            let colors = [
                UIColor(white: 0, alpha: 0.0).cgColor,
                UIColor(white: 0, alpha: 1.0).cgColor
            ] as CFArray
            
            cachedGradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceRGB(),
                colors: colors,
                locations: [0.4, 1.0]
            )
        }
        let gradientTime = Date().timeIntervalSince(gradientStartTime) * 1000
        print("⏱ Gradient creation time: \(gradientTime)ms")
        
        // Step 6: Apply gradient mask and blend lips
        let blendStartTime = Date()
        guard let predictedCGImage = predictedLips.cgImage,
              let gradient = cachedGradient else { return nil }
        
        context?.saveGState()
        context?.addEllipse(in: lipRect)
        context?.clip()
        
        context?.drawLinearGradient(
            gradient,
            start: CGPoint(x: lipRect.midX, y: lipRect.minY),
            end: CGPoint(x: lipRect.midX, y: lipRect.maxY),
            options: []
        )
        
        context?.setBlendMode(.overlay)
        context?.draw(predictedCGImage, in: CGRect(origin: .zero, size: targetSize))
        context?.restoreGState()
        
        let blendTime = Date().timeIntervalSince(blendStartTime) * 1000
        print("⏱ Blending time: \(blendTime)ms")
        
        // Step 7: Create final image
        let finalStartTime = Date()
        guard let provider = CGDataProvider(data: Data(pixelData) as CFData),
              let cgImage = CGImage(
                width: Int(targetSize.width),
                height: Int(targetSize.height),
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: Int(targetSize.width * 4),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            print("❌ Failed to create final image")
            return nil
        }
        let finalTime = Date().timeIntervalSince(finalStartTime) * 1000
        print("⏱ Final image creation time: \(finalTime)ms")
        
        let totalTime = Date().timeIntervalSince(startTime) * 1000
        print("\n📊 Lip Merging Performance Summary:")
        print("Total time: \(totalTime)ms")
        print("\nTime breakdown:")
        print("- Original face conversion: \(originalTime)ms (\(originalTime/totalTime*100)%)")
        print("- Buffer preparation: \(bufferTime)ms (\(bufferTime/totalTime*100)%)")
        print("- Gradient creation: \(gradientTime)ms (\(gradientTime/totalTime*100)%)")
        print("- Blending: \(blendTime)ms (\(blendTime/totalTime*100)%)")
        print("- Final image creation: \(finalTime)ms (\(finalTime/totalTime*100)%)")
        
        return UIImage(cgImage: cgImage)
    }

    
    // Add helper method for debugging
    private func debugOutputArray(_ array: MLMultiArray, tag: String = "") {
        var minValue: Float = 1.0
        var maxValue: Float = 0.0
        var sum: Float = 0.0
        
        for i in 0..<array.count {
            let value = array[i].floatValue
            minValue = min(minValue, value)
            maxValue = max(maxValue, value)
            sum += value
        }
        
        let mean = sum / Float(array.count)
        print("🔍 \(tag) Array Stats:")
        print("- Shape: \(array.shape.map { $0.intValue })")
        print("- Range: [\(minValue), \(maxValue)]")
        print("- Mean: \(mean)")
    }
    
    func generateVideo() -> URL? {
        guard !generatedFrames.isEmpty else {
            print("❌ No frames available")
            return nil
        }
        
        print("🎬 Generating video from \(generatedFrames.count) frames...")
        return videoGenerator.createVideo(from: generatedFrames)
    }
    
    func clearFrames() {
        generatedFrames.removeAll()
    }
    
    // Debug flag for development
    private let isDebugMode = false
}

