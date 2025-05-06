import Vision
import UIKit
import CoreML
import Foundation


class FaceProcessor {
    
    // Constants for face processing
    private let targetSize = CGSize(width: 96, height: 96)
    private let sequenceLength = 5  // Number of frames needed
    private let meanPixelValue: Float = 0.5
    private let pixelScale: Float = 0.5
    
    // Add new constants for lips region
    private let lipsVerticalOffset: CGFloat = 0.55  // Position of lips from top of face
    private let lipsHeightRatio: CGFloat = 0.35    // Height of lips relative to face
    
    // Face detection request
    private lazy var faceDetectionRequest: VNDetectFaceRectanglesRequest = {
        let request = VNDetectFaceRectanglesRequest()
        request.preferBackgroundProcessing = true
        return request
    }()
    
    private let lipsRegion = (
        verticalOffset: 0.5,  // Start at middle of face
        height: 0.4          // Cover 40% of face height
    )
    
    func validateFaceSequence(_ array: MLMultiArray) -> Bool {
        print("🔍 Validating face sequence...")
        
        // Check shape
        let expectedShape = [1, 6, 96, 96]
        let actualShape = array.shape.map { $0.intValue }
        guard actualShape == expectedShape else {
            print("❌ Shape mismatch: expected \(expectedShape), got \(actualShape)")
            return false
        }
        print("✅ Shape validation passed")
        
        // Check value ranges
        var minValue: Float = 1.0
        var maxValue: Float = 0.0
        
        for i in 0..<array.count {
            let value = array[i].floatValue
            minValue = min(minValue, value)
            maxValue = max(maxValue, value)
            
            if value < 0.0 || value > 1.0 {
                print("❌ Value out of range [0,1]: \(value) at index \(i)")
                return false
            }
        }
        
        print("✅ Value range validation passed: min=\(minValue), max=\(maxValue)")
        return true
    }
    func extractLipRegion(from image: UIImage) -> UIImage? {
        print("👄 Extracting lip region...")
        
        // Detect face
        guard let face = detectFaceInternal(in: image) else {
            print("❌ No face detected")
            return nil
        }
        
        // Get image dimensions
        guard let cgImage = image.cgImage else { return nil }
        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
        
        // Calculate face bounds
        let faceBox = face.boundingBox
        let faceX = faceBox.origin.x * imageSize.width
        let faceY = (1 - faceBox.maxY) * imageSize.height
        let faceWidth = faceBox.width * imageSize.width
        let faceHeight = faceBox.height * imageSize.height
        
        // Calculate lip region within face
        // Typically, lips are in the lower third of the face
        let lipYOffset = faceHeight * 0.55  // Start at 55% from the top of face
        let lipHeight = faceHeight * 0.28   // Cover about 28% of face height
        
        let lipRect = CGRect(
            x: faceX + faceWidth * 0.25,           // Center horizontally (25% from left)
            y: faceY + lipYOffset,                 // Positioned vertically
            width: faceWidth * 0.5,                // 50% of face width
            height: lipHeight                      // Height as calculated
        )
        
        print("👄 Lip region calculated at: \(lipRect)")
        
        // Ensure the lip region is valid
        guard lipRect.width > 0, lipRect.height > 0,
              lipRect.maxX <= imageSize.width, lipRect.maxY <= imageSize.height else {
            print("❌ Invalid lip region bounds")
            return nil
        }
        
        // Crop lip region
        guard let croppedCGImage = cgImage.cropping(to: lipRect) else {
            print("❌ Failed to crop lip region")
            return nil
        }
        
        // Create lip region image
        let lipImage = UIImage(cgImage: croppedCGImage)
        
        // Resize to target size if needed for the model
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        
        let resizedLips = UIGraphicsImageRenderer(size: targetSize, format: format).image { context in
            context.cgContext.interpolationQuality = .high
            lipImage.draw(in: CGRect(origin: .zero, size: targetSize))
        }
        
        print("✅ Lip region extracted successfully")
        return resizedLips
    }
    
    func extractFaceSequenceForLipSync(from image: UIImage) -> (sequence: MLMultiArray?, originalImage: UIImage, lipRegion: CGRect?) {
        print("🔍 Processing face for lip sync...")
        
        // Store original image
        let originalImage = image
        
        // Detect face
        guard let face = detectFaceInternal(in: image) else {
            print("❌ No face detected")
            return (nil, originalImage, nil)
        }
        
        // Calculate face and lip regions
        guard let cgImage = image.cgImage else {
            return (nil, originalImage, nil)
        }
        
        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
        let faceBox = face.boundingBox
        
        // Calculate face bounds
        let faceX = faceBox.origin.x * imageSize.width
        let faceY = (1 - faceBox.maxY) * imageSize.height
        let faceWidth = faceBox.width * imageSize.width
        let faceHeight = faceBox.height * imageSize.height
        
        // Calculate lip region with widened parameters
        let lipYOffset = faceHeight * 0.58  // Keep lower position
        let lipHeight = faceHeight * 0.22   // Keep tighter fit
        let lipWidth = faceWidth * 0.50     // Increase to 50% for full mouth coverage
        
        let lipRect = CGRect(
            x: faceX + faceWidth * 0.30,  // Shift slightly left
            y: faceY + lipYOffset,
            width: lipWidth,
            height: lipHeight
        )
        
        // Process face for model input
        guard let processedFace = cropAndResizeFace(image: image, face: face, targetSize: targetSize),
              let maskedFace = applyLowerHalfMask(to: processedFace) else {
            print("❌ Face processing failed")
            return (nil, originalImage, lipRect)
        }
        do {
            let shape: [NSNumber] = [5, 6, 96, 96]
            let mlArray = try MLMultiArray(shape: shape, dataType: .float32)
            
            // 转换processedFace到像素缓冲区
            guard let processedPixelBuffer = processedFace.toPixelBuffer() else {
                print("❌ Failed to create processed face pixel buffer")
                return (nil, originalImage, lipRect)
            }
            
            // 转换maskedFace到像素缓冲区
            guard let maskedPixelBuffer = maskedFace.toPixelBuffer() else {
                print("❌ Failed to create masked face pixel buffer")
                return (nil, originalImage, lipRect)
            }
            
            // 锁定并访问processedFace像素数据
            CVPixelBufferLockBaseAddress(processedPixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(processedPixelBuffer, .readOnly) }
            
            guard let processedBaseAddress = CVPixelBufferGetBaseAddress(processedPixelBuffer) else {
                print("❌ Failed to get processed pixel buffer base address")
                return (nil, originalImage, lipRect)
            }
            
            let processedBytesPerRow = CVPixelBufferGetBytesPerRow(processedPixelBuffer)
            let processedBuffer = processedBaseAddress.assumingMemoryBound(to: UInt8.self)
            
            // 锁定并访问maskedFace像素数据
            CVPixelBufferLockBaseAddress(maskedPixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(maskedPixelBuffer, .readOnly) }
            
            guard let maskedBaseAddress = CVPixelBufferGetBaseAddress(maskedPixelBuffer) else {
                print("❌ Failed to get masked pixel buffer base address")
                return (nil, originalImage, lipRect)
            }
            
            let maskedBytesPerRow = CVPixelBufferGetBytesPerRow(maskedPixelBuffer)
            let maskedBuffer = maskedBaseAddress.assumingMemoryBound(to: UInt8.self)
            
            // 填充MLMultiArray
            for frame in 0..<5 {
                for y in 0..<96 {
                    for x in 0..<96 {
                        // 计算processedFace中像素的偏移量
                        let processedOffset = y * processedBytesPerRow + x * 4
                        // 计算maskedFace中像素的偏移量
                        let maskedOffset = y * maskedBytesPerRow + x * 4
                        
                        // 获取processedFace的RGB值
                        let processedR = Float(processedBuffer[processedOffset]) / 255.0
                        let processedG = Float(processedBuffer[processedOffset + 1]) / 255.0
                        let processedB = Float(processedBuffer[processedOffset + 2]) / 255.0
                        
                        // 获取maskedFace的RGB值
                        let maskedR = Float(maskedBuffer[maskedOffset]) / 255.0
                        let maskedG = Float(maskedBuffer[maskedOffset + 1]) / 255.0
                        let maskedB = Float(maskedBuffer[maskedOffset + 2]) / 255.0
                        
                        // 前3个通道使用processedFace的RGB
                        mlArray[[frame, 0, y, x] as [NSNumber]] = NSNumber(value: maskedR)
                        mlArray[[frame, 1, y, x] as [NSNumber]] = NSNumber(value: maskedG)
                        mlArray[[frame, 2, y, x] as [NSNumber]] = NSNumber(value: maskedB)
                        
                        // 后3个通道使用maskedFace的RGB
                        mlArray[[frame, 3, y, x] as [NSNumber]] = NSNumber(value: processedR)
                        mlArray[[frame, 4, y, x] as [NSNumber]] = NSNumber(value: processedG)
                        mlArray[[frame, 5, y, x] as [NSNumber]] = NSNumber(value: processedB)
                    }
                }
            }
            
            print("✅ Face sequence created successfully with processedFace and maskedFace")
            return (mlArray, originalImage, lipRect)
            
        } catch {
            print("❌ Error creating face sequence: \(error)")
            return (nil, originalImage, lipRect)
        }

        // Create MLMultiArray for model input
//        do {
//            let shape: [NSNumber] = [5, 6, 96, 96]
//            let mlArray = try MLMultiArray(shape: shape, dataType: .float32)
//            
//            // Convert masked face to pixel buffer
//            guard let pixelBuffer = maskedFace.toPixelBuffer() else {
//                print("❌ Failed to create pixel buffer")
//                return (nil, originalImage, lipRect)
//            }
//            
//            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
//            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
//            
//            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
//                print("❌ Failed to get pixel buffer base address")
//                return (nil, originalImage, lipRect)
//            }
//            
//            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
//            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
//            
//            // Fill all 5 frames with the same masked face
//            for frame in 0..<5 {
//                for y in 0..<96 {
//                    for x in 0..<96 {
//                        let offset = y * bytesPerRow + x * 4
//                        let r = Float(buffer[offset]) / 255.0
//                        let g = Float(buffer[offset + 1]) / 255.0
//                        let b = Float(buffer[offset + 2]) / 255.0
//                        
//                        mlArray[[frame, 0, y, x] as [NSNumber]] = NSNumber(value: r)
//                        mlArray[[frame, 1, y, x] as [NSNumber]] = NSNumber(value: g)
//                        mlArray[[frame, 2, y, x] as [NSNumber]] = NSNumber(value: b)
//                        mlArray[[frame, 3, y, x] as [NSNumber]] = NSNumber(value: b)
//                        mlArray[[frame, 4, y, x] as [NSNumber]] = NSNumber(value: g)
//                        mlArray[[frame, 5, y, x] as [NSNumber]] = NSNumber(value: r)
//                    }
//                }
//            }
//            
//            print("✅ Face sequence created successfully")
//            return (mlArray, originalImage, lipRect)
//            
//        } catch {
//            print("❌ Error creating face sequence: \(error)")
//            return (nil, originalImage, lipRect)
//        }
    }
    
    
    func extractFaceSequence(from image: UIImage) -> MLMultiArray? {
        print("🔍 Processing face image for sequence...")
        
        do {
            let shape: [NSNumber] = [5, 6, 96, 96]
            let mlArray = try MLMultiArray(shape: shape, dataType: .float16)
            
            // Process face
            guard let face = detectFaceInternal(in: image),
                  let processedFace = cropAndResizeFace(image: image, face: face, targetSize: targetSize),
                  let maskedFace = applyLowerHalfMask(to: processedFace) else {
                print("❌ Face processing failed")
                return nil
            }
            
            // Get pixel data from both faces
            guard let originalBuffer = processedFace.toPixelBuffer(),
                  let maskedBuffer = maskedFace.toPixelBuffer() else {
                print("❌ Failed to create pixel buffers")
                return nil
            }
            
            // Lock buffers
            CVPixelBufferLockBaseAddress(originalBuffer, .readOnly)
            CVPixelBufferLockBaseAddress(maskedBuffer, .readOnly)
            defer {
                CVPixelBufferUnlockBaseAddress(originalBuffer, .readOnly)
                CVPixelBufferUnlockBaseAddress(maskedBuffer, .readOnly)
            }
            
            // Get base addresses
            guard let originalBaseAddress = CVPixelBufferGetBaseAddress(originalBuffer),
                  let maskedBaseAddress = CVPixelBufferGetBaseAddress(maskedBuffer) else {
                print("❌ Failed to get pixel buffer base addresses")
                return nil
            }
            
            // Use different variable names to avoid redeclaration
            let originalPixels = originalBaseAddress.assumingMemoryBound(to: UInt8.self)
            let maskedPixels = maskedBaseAddress.assumingMemoryBound(to: UInt8.self)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(originalBuffer)
            
            // Fill MLMultiArray
            for frame in 0..<5 {
                for y in 0..<96 {
                    for x in 0..<96 {
                        let offset = y * bytesPerRow + x * 4
                        
                        // Get RGB values from original face
                        let originalR = Float(originalPixels[offset + 1]) / 255.0
                        let originalG = Float(originalPixels[offset + 2]) / 255.0
                        let originalB = Float(originalPixels[offset + 3]) / 255.0
                        
                        // Get RGB values from masked face
                        let maskedR = Float(maskedPixels[offset + 1]) / 255.0
                        let maskedG = Float(maskedPixels[offset + 2]) / 255.0
                        let maskedB = Float(maskedPixels[offset + 3]) / 255.0
                        
                        // Fill original face channels (0-2)
                        mlArray[[frame, 0, y, x] as [NSNumber]] = NSNumber(value: originalR)
                        mlArray[[frame, 1, y, x] as [NSNumber]] = NSNumber(value: originalG)
                        mlArray[[frame, 2, y, x] as [NSNumber]] = NSNumber(value: originalB)
                        
                        // Fill masked face channels (3-5)
                        mlArray[[frame, 3, y, x] as [NSNumber]] = NSNumber(value: maskedR)
                        mlArray[[frame, 4, y, x] as [NSNumber]] = NSNumber(value: maskedG)
                        mlArray[[frame, 5, y, x] as [NSNumber]] = NSNumber(value: maskedB)
                    }
                }
            }
            
            print("✅ Face sequence created successfully")
            return mlArray
            
        } catch {
            print("❌ Error creating face sequence: \(error)")
            return nil
        }
    }
    
    public func detectFaceInternal(in image: UIImage) -> VNFaceObservation? {
        guard let cgImage = image.cgImage else { return nil }
        
        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        do {
            try handler.perform([request])
            if let firstFace = request.results?.first as? VNFaceObservation {
                print("✅ Face detected at: \(firstFace.boundingBox)")
                print("📏 Face bounding box size: \(firstFace.boundingBox.size)")
                return firstFace
            } else {
                print("❌ No face detected")
                return nil
            }
        } catch {
            print("❌ Face detection failed: \(error)")
            return nil
        }
    }
    
    func cropAndResizeFace(image: UIImage, face: VNFaceObservation, targetSize: CGSize) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        // Get image dimensions
        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
        
        // Convert normalized coordinates to pixel coordinates (Y-Axis correction)
        let faceBox = face.boundingBox
        let x = faceBox.origin.x * imageSize.width
        let y = (1 - faceBox.maxY) * imageSize.height // ✅ Corrected Y-Axis transformation
        let width = faceBox.width * imageSize.width
        let height = faceBox.height * imageSize.height

        // ✅ Ensure cropping rectangle stays within image bounds
        let faceRect = CGRect(
            x: max(0, x),
            y: max(0, y),
            width: min(width, imageSize.width - x),
            height: min(height, imageSize.height - y)
        )

        print("🔍 Face crop bounds (updated): \(faceRect)")

        // ✅ Validate Cropping Bounds
        guard faceRect.width > 0, faceRect.height > 0 else {
            print("❌ Invalid Face Crop Dimensions")
            return nil
        }
        
        // ✅ Perform cropping safely
        guard let croppedCGImage = cgImage.cropping(to: faceRect) else {
            print("❌ Face Cropping Failed: Invalid bounds")
            return nil
        }

        // ✅ Preserve aspect ratio when resizing
        let aspectRatio = faceRect.width / faceRect.height
        let newSize: CGSize
        if aspectRatio > 1 {
            // Wider face
            newSize = CGSize(width: targetSize.width, height: targetSize.width / aspectRatio)
        } else {
            // Taller face
            newSize = CGSize(width: targetSize.height * aspectRatio, height: targetSize.height)
        }

        // ✅ Create resized image without distortion
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        
        let resizedImage = UIGraphicsImageRenderer(size: targetSize, format: format).image { context in
            context.cgContext.interpolationQuality = .high
            UIImage(cgImage: croppedCGImage).draw(in: CGRect(x: (targetSize.width - newSize.width) / 2,
                                                             y: (targetSize.height - newSize.height) / 2,
                                                             width: newSize.width,
                                                             height: newSize.height))
        }

        return resizedImage
    }

    func debugFaceProcessing(image: UIImage) -> UIImage? {
        guard let cgImage = image.cgImage else {
            print("❌ Could not get CGImage")
            return nil
        }
        
        guard let face = detectFaceInternal(in: image) else {
            print("❌ No face detected")
            return nil
        }
        
        let size = CGSize(width: cgImage.width, height: cgImage.height)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        let debugImage = renderer.image { context in
            // Draw original image
            image.draw(in: CGRect(origin: .zero, size: size))
            
            // Draw face rectangle with thicker red outline
            let bounds = VNImageRectForNormalizedRect(face.boundingBox,
                                                     Int(size.width),
                                                     Int(size.height))
            
            context.cgContext.setStrokeColor(UIColor.red.cgColor)
            context.cgContext.setLineWidth(4)
            context.cgContext.stroke(bounds)
            
            // Draw processed face preview with green outline
            if let processedFace = cropAndResizeFace(image: image,
                                                    face: face,
                                                    targetSize: targetSize) {
                let previewSize = CGSize(width: 96, height: 96)
                let previewRect = CGRect(x: 10, y: 10, width: previewSize.width, height: previewSize.height)
                processedFace.draw(in: previewRect)
                
                context.cgContext.setStrokeColor(UIColor.green.cgColor)
                context.cgContext.setLineWidth(2)
                context.cgContext.stroke(previewRect)
                
                // Add debug info
                let debugText = "Face detected at: \(face.boundingBox)"
                let attributes: [NSAttributedString.Key: Any] = [
                    .foregroundColor: UIColor.yellow,
                    .font: UIFont.systemFont(ofSize: 12)
                ]
                debugText.draw(at: CGPoint(x: 10, y: 116), withAttributes: attributes)
            }
        }
        
        return debugImage
    }
    
    func testFaceProcessing(image: UIImage) -> [String: Any] {
        var results = [String: Any]()
        
        // Step 1: Test image loading
        guard image.cgImage != nil else {
            print("❌ Failed: Could not get CGImage")
            results["status"] = "failed"
            results["error"] = "Image loading failed"
            return results
        }
        
        // Step 2: Test face detection
        guard let face = detectFaceInternal(in: image) else {
            print("❌ Failed: No face detected in image")
            results["status"] = "failed"
            results["error"] = "Face detection failed"
            return results
        }
        
        // Step 3: Test cropping and resizing
        guard let processedFace = cropAndResizeFace(image: image,
                                                   face: face,
                                                   targetSize: targetSize) else {
            print("❌ Failed: Could not crop and resize face")
            results["status"] = "failed"
            results["error"] = "Face processing failed"
            return results
        }
        
        print("✅ Face cropped and resized to: \(targetSize)")
        results["processedSize"] = "\(targetSize)"
        
        // Step 4: Test normalization
        guard let normalizedPixels = processedFace.toPixelBuffer() else {
            print("❌ Failed: Could not normalize pixels")
            results["status"] = "failed"
            results["error"] = "Normalization failed"
            return results
        }
        
        // Step 5: Test sequence generation
        do {
            let shape: [NSNumber] = [5, 6, 96, 96]
            let mlArray = try MLMultiArray(shape: shape, dataType: .float16)
            
            // Process pixel values
            CVPixelBufferLockBaseAddress(normalizedPixels, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(normalizedPixels, .readOnly) }
            
            guard let baseAddress = CVPixelBufferGetBaseAddress(normalizedPixels) else {
                throw NSError(domain: "FaceProcessor", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to get pixel buffer base address"])
            }
            
            let bytesPerRow = CVPixelBufferGetBytesPerRow(normalizedPixels)
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            
            // Fill MLMultiArray
            for batch in 0..<5 {
                for y in 0..<96 {
                    for x in 0..<96 {
                        let offset = y * bytesPerRow + x * 4
                        
                        let r = Float(buffer[offset + 0]) / 255.0
                        let g = Float(buffer[offset + 1]) / 255.0
                        let b = Float(buffer[offset + 2]) / 255.0
                        
                        let normalizedR = (r - meanPixelValue) / pixelScale
                        let normalizedG = (g - meanPixelValue) / pixelScale
                        let normalizedB = (b - meanPixelValue) / pixelScale
                        
                        mlArray[[batch, 0, y, x] as [NSNumber]] = NSNumber(value: normalizedR)
                        mlArray[[batch, 1, y, x] as [NSNumber]] = NSNumber(value: normalizedG)
                        mlArray[[batch, 2, y, x] as [NSNumber]] = NSNumber(value: normalizedB)
                        mlArray[[batch, 3, y, x] as [NSNumber]] = NSNumber(value: normalizedB)
                        mlArray[[batch, 4, y, x] as [NSNumber]] = NSNumber(value: normalizedG)
                        mlArray[[batch, 5, y, x] as [NSNumber]] = NSNumber(value: normalizedR)
                    }
                }
            }
            
            print("✅ Face sequence created with shape: \(mlArray.shape.map { $0.intValue })")
            results["status"] = "success"
            results["shape"] = mlArray.shape.map { $0.intValue }
            
        } catch {
            print("❌ Failed: Could not create face sequence")
            results["status"] = "failed"
            results["error"] = error.localizedDescription
        }
        
        return results
    }
    
    func processStaticImage(_ image: UIImage) -> (face: MLMultiArray?, lips: UIImage?)? {
        print("🔍 Processing static image for lip sync...")
        
        guard let face = detectFaceInternal(in: image) else {
            print("❌ No face detected in the static image")
            return nil
        }
        
        // Get both the full face and lips region
        guard let processedFace = cropAndResizeFace(image: image, face: face, targetSize: targetSize),
              let lipsRegion = cropLipsRegion(image: image, face: face, targetSize: targetSize) else {
            print("❌ Failed to process face regions")
            return nil
        }
        
        // Convert face to MLMultiArray for the model
        if let faceArray = convertToMLArray(from: processedFace) {
            return (face: faceArray, lips: lipsRegion)
        }
        
        return nil
    }
    
    private func cropLipsRegion(image: UIImage, face: VNFaceObservation, targetSize: CGSize) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        
        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
        let faceBox = face.boundingBox
        
        // Calculate lips region
        let x = faceBox.origin.x * imageSize.width
        let y = (1 - faceBox.origin.y - faceBox.height) * imageSize.height
        let width = faceBox.width * imageSize.width
        let height = faceBox.height * imageSize.height
        
        // Focus on lips area
        let lipsY = y + (height * 0.6) // 60% from the top of the face
        let lipsHeight = height * 0.3 // Covering 30% of the face from lower part

        
        let lipsRect = CGRect(
            x: max(0, x),
            y: max(0, lipsY),
            width: min(width, imageSize.width - x),
            height: min(lipsHeight, imageSize.height - lipsY)
        )
        
        print("🔍 Lips region bounds: \(lipsRect)")
        
        guard let croppedLips = cgImage.cropping(to: lipsRect) else {
            print("❌ Failed to crop lips region")
            return nil
        }
        
        // Resize lips region
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        
        let resizedLips = UIGraphicsImageRenderer(size: targetSize, format: format).image { context in
            context.cgContext.interpolationQuality = .high
            UIImage(cgImage: croppedLips).draw(in: CGRect(origin: .zero, size: targetSize))
        }
        
        return resizedLips
    }
    
    private func convertToMLArray(from image: UIImage) -> MLMultiArray? {
        do {
            let shape: [NSNumber] = [5, 6, 96, 96]
            let mlArray = try MLMultiArray(shape: shape, dataType: .float16)
            
            guard let pixelBuffer = image.toPixelBuffer() else {
                print("❌ Failed to create pixel buffer")
                return nil
            }
            
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
            
            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                print("❌ Failed to get pixel buffer base address")
                return nil
            }
            
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            
            // Fill array with normalized values
            for batch in 0..<5 {
                for y in 0..<96 {
                    for x in 0..<96 {
                        let offset = y * bytesPerRow + x * 4
                        
                        let r = Float(buffer[offset + 0]) / 255.0
                        let g = Float(buffer[offset + 1]) / 255.0
                        let b = Float(buffer[offset + 2]) / 255.0
                        
                        let normalizedR = max(0, min(1, (r - meanPixelValue) / pixelScale))
                        let normalizedG = max(0, min(1, (g - meanPixelValue) / pixelScale))
                        let normalizedB = max(0, min(1, (b - meanPixelValue) / pixelScale))
                        
                        mlArray[[batch, 0, y, x] as [NSNumber]] = NSNumber(value: normalizedR)
                        mlArray[[batch, 1, y, x] as [NSNumber]] = NSNumber(value: normalizedG)
                        mlArray[[batch, 2, y, x] as [NSNumber]] = NSNumber(value: normalizedB)
                        mlArray[[batch, 3, y, x] as [NSNumber]] = NSNumber(value: normalizedB)
                        mlArray[[batch, 4, y, x] as [NSNumber]] = NSNumber(value: normalizedG)
                        mlArray[[batch, 5, y, x] as [NSNumber]] = NSNumber(value: normalizedR)
                    }
                }
            }
            
            return mlArray
            
        } catch {
            print("❌ Error creating MLMultiArray: \(error)")
            return nil
        }
    }
    
    func loadStaticFace(named imageName: String = "Elon") -> MLMultiArray? {
        print("🔍 Loading static face image: \(imageName)")
        
        // Look for image in Preview Assets catalog
        guard let image = UIImage(named: imageName, in: Bundle.main, compatibleWith: nil) else {
            print("❌ Failed to load image: \(imageName) from Preview Assets")
            return nil
        }
        
        return processFace(image: image)
    }
    
    private func processFace(image: UIImage) -> MLMultiArray? {
        guard let face = detectFaceInternal(in: image) else {
            print("❌ No face detected in image")
            return nil
        }
        
        print("✅ Face detected at: \(face.boundingBox)")
        
        guard let croppedFace = cropFaceRegion(image: image, face: face) else {
            print("❌ Failed to crop face region")
            return nil
        }
        
        guard let faceArray = convertToMLMultiArray(from: croppedFace) else {
            print("❌ Failed to convert face to MLMultiArray")
            return nil
        }
        
        print("✅ Face processed successfully")
        return faceArray
    }
    
    private func cropFaceRegion(image: UIImage, face: VNFaceObservation) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        
        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
        let boundingBox = face.boundingBox
        
        // Convert normalized coordinates to pixel coordinates
        let x = boundingBox.origin.x * imageSize.width
        let y = (1 - boundingBox.origin.y - boundingBox.height) * imageSize.height
        let width = boundingBox.width * imageSize.width
        let height = boundingBox.height * imageSize.height
        
        // Add padding around face
        let paddingX = width * 0.1
        let paddingY = height * 0.1
        
        let faceRect = CGRect(
            x: max(0, x - paddingX),
            y: max(0, y - paddingY),
            width: min(width + (paddingX * 2), imageSize.width - x + paddingX),
            height: min(height + (paddingY * 2), imageSize.height - y + paddingY)
        )
        
        guard let croppedImage = cgImage.cropping(to: faceRect) else {
            print("❌ Failed to crop face region")
            return nil
        }
        
        // Resize to target size
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        
        let resizedImage = UIGraphicsImageRenderer(size: targetSize, format: format).image { context in
            context.cgContext.interpolationQuality = .high
            UIImage(cgImage: croppedImage).draw(in: CGRect(origin: .zero, size: targetSize))
        }
        
        return resizedImage
    }
    
    private func convertToMLMultiArray(from image: UIImage) -> MLMultiArray? {
        print("🔄 Converting image to MLMultiArray...")
        
        do {
            let shape: [NSNumber] = [1, 6, 96, 96] // Shape expected by Wav2Lip
            let mlArray = try MLMultiArray(shape: shape, dataType: .float32)
            
            guard let pixelBuffer = image.toPixelBuffer() else {
                print("❌ Failed to create pixel buffer")
                return nil
            }
            
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
            
            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                print("❌ Failed to get pixel buffer base address")
                return nil
            }
            
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            
            // Track value ranges for debugging
            var minValue: Float = 1.0
            var maxValue: Float = 0.0
            
            // Convert pixels into MLMultiArray format
            for y in 0..<96 {
                for x in 0..<96 {
                    let offset = y * bytesPerRow + x * 4
                    let r = Float(buffer[offset]) / 255.0
                    let g = Float(buffer[offset + 1]) / 255.0
                    let b = Float(buffer[offset + 2]) / 255.0
                    
                    // Track value ranges
                    minValue = min(minValue, min(r, min(g, b)))
                    maxValue = max(maxValue, max(r, max(g, b)))
                    
                    // Store RGB values in all 6 channels
                    mlArray[[0, 0, y, x] as [NSNumber]] = NSNumber(value: r)
                    mlArray[[0, 1, y, x] as [NSNumber]] = NSNumber(value: g)
                    mlArray[[0, 2, y, x] as [NSNumber]] = NSNumber(value: b)
                    mlArray[[0, 3, y, x] as [NSNumber]] = NSNumber(value: b)
                    mlArray[[0, 4, y, x] as [NSNumber]] = NSNumber(value: g)
                    mlArray[[0, 5, y, x] as [NSNumber]] = NSNumber(value: r)
                }
            }
            
            print("✅ Conversion complete")
            print("📊 Value range: min=\(minValue), max=\(maxValue)")
            return mlArray
            
        } catch {
            print("❌ Error creating MLMultiArray: \(error)")
            return nil
        }
    }
    


    /// Extracts face sequence correctly formatted for Wav2Lip (5, 6, 96, 96)
    func extractMaskedFace(from image: UIImage) -> MLMultiArray? {
        print("🔍 Processing face image for sequence...")

        let startTime = Date()  // Start time tracking

        do {
            // Create output array with shape [5, 6, 96, 96]
            let shape: [NSNumber] = [5, 6, 96, 96]
            let mlArray = try MLMultiArray(shape: shape, dataType: .float32)

            // Detect and process face
            guard let face = detectFaceInternal(in: image),
                  let processedFace = cropAndResizeFace(image: image, face: face, targetSize: targetSize),
                  let maskedFace = applyLowerHalfMask(to: processedFace) else {
                print("❌ Face processing failed")
                return nil
            }

            print("✅ Face detected and masked")

            // Debugging: Save processed face and masked face for verification
            if debugFaceProcessing(image: processedFace) != nil {
                print("🖼️ Saving debug face image for verification")
            }

            // Convert masked face to pixel buffer
            guard let pixelBuffer = maskedFace.toPixelBuffer() else {
                print("❌ Failed to create pixel buffer")
                return nil
            }

            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                print("❌ Failed to get pixel buffer base address")
                return nil
            }

            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)

            let frameStartTime = Date() // Start tracking time for frame processing

            // Fill all 5 frames with the same masked face
            for frame in 0..<5 {
                for y in 0..<96 {
                    for x in 0..<96 {
                        let offset = y * bytesPerRow + x * 4
                        let r = Float(buffer[offset]) / 255.0
                        let g = Float(buffer[offset + 1]) / 255.0
                        let b = Float(buffer[offset + 2]) / 255.0

                        mlArray[[frame, 0, y, x] as [NSNumber]] = NSNumber(value: r)
                        mlArray[[frame, 1, y, x] as [NSNumber]] = NSNumber(value: g)
                        mlArray[[frame, 2, y, x] as [NSNumber]] = NSNumber(value: b)
                        mlArray[[frame, 3, y, x] as [NSNumber]] = NSNumber(value: b)
                        mlArray[[frame, 4, y, x] as [NSNumber]] = NSNumber(value: g)
                        mlArray[[frame, 5, y, x] as [NSNumber]] = NSNumber(value: r)
                    }
                }
            }

            let frameEndTime = Date() // End time for frame processing
            let frameProcessingTime = frameEndTime.timeIntervalSince(frameStartTime) * 1000 // Convert to ms
            print("⏱ Face frame processing time: \(frameProcessingTime) ms")

            let endTime = Date()  // End time tracking
            let totalProcessingTime = endTime.timeIntervalSince(startTime) * 1000 // Convert to ms
            print("✅ Face sequence created with shape: \(mlArray.shape.map { $0.intValue })")
            print("⏱ Total face processing time: \(totalProcessingTime) ms")

            return mlArray

        } catch {
            print("❌ Error creating face sequence: \(error)")
            return nil
        }
    }

    
//    private func applyLowerHalfMask(to image: UIImage) -> UIImage? {
//        let renderer = UIGraphicsImageRenderer(size: targetSize)
//        
//        return renderer.image { context in
//            // Draw original image
//            image.draw(in: CGRect(origin: .zero, size: targetSize))
//            
//            // Calculate lip region - focus on lower third of face
//            let lipRegionStart = targetSize.height * 0.6  // Start at 60% from top
//            let lipRegionHeight = targetSize.height * 0.25 // Cover 25% of face height
//            
//            // Create mask for lip region with gradient edges
//            let lipRect = CGRect(
//                x: targetSize.width * 0.25, // Start at 25% from left
//                y: lipRegionStart,
//                width: targetSize.width * 0.5, // Cover 50% of face width
//                height: lipRegionHeight
//            )
//            
//            // Create gradient for smooth blending
//            let gradient = CGGradient(
//                colorsSpace: CGColorSpaceCreateDeviceRGB(),
//                colors: [
//                    UIColor.black.withAlphaComponent(0.8).cgColor,
//                    UIColor.black.withAlphaComponent(0.5).cgColor
//                ] as CFArray,
//                locations: [0, 1]
//            )!
//            
//            // Apply gradient mask
//            context.cgContext.addRect(lipRect)
//            context.cgContext.clip()
//            
//            // Draw gradient
//            context.cgContext.drawLinearGradient(
//                gradient,
//                start: CGPoint(x: lipRect.midX, y: lipRect.minY),
//                end: CGPoint(x: lipRect.midX, y: lipRect.maxY),
//                options: []
//            )
//        }
//    }
//
    
    private func applyLowerHalfMask(to image: UIImage) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        
        return renderer.image { context in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
            
            let bottomHalfRect = CGRect(
                x: 0,
                y: targetSize.height / 2,
                width: targetSize.width,
                height: targetSize.height / 2
            )
            
            UIColor.black.setFill()
            
            context.fill(bottomHalfRect)
        }
    }
    
    // Debug method to visualize face detection
    func debugFaceDetection(in image: UIImage) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        do {
            try handler.perform([faceDetectionRequest])
            
            let renderer = UIGraphicsImageRenderer(size: image.size)
            return renderer.image { context in
                // Draw original image
                image.draw(in: CGRect(origin: .zero, size: image.size))
                
                // Draw face rectangles
                context.cgContext.setStrokeColor(UIColor.red.cgColor)
                context.cgContext.setLineWidth(3)
                
                faceDetectionRequest.results?.forEach { observation in
                    if let face = observation as? VNFaceObservation {
                        let box = face.boundingBox
                        let rect = CGRect(
                            x: box.origin.x * image.size.width,
                            y: (1 - box.origin.y - box.height) * image.size.height,
                            width: box.width * image.size.width,
                            height: box.height * image.size.height
                        )
                        context.cgContext.stroke(rect)
                    }
                }
            }
        } catch {
            print("❌ Debug visualization failed: \(error)")
            return nil
        }
    }
}

extension UIImage {
    func toPixelBuffer() -> CVPixelBuffer? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attributes,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )
        
        context?.draw(cgImage!, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
    
    // Helper method to create MLMultiArray from pixel data
    func toMLMultiArray(shape: [NSNumber]) -> MLMultiArray? {
        guard let pixelBuffer = self.toPixelBuffer() else {
            print("❌ Failed to get pixel buffer")
            return nil
        }
        
        do {
            let mlArray = try MLMultiArray(shape: shape, dataType: .float16)
            
            // Lock buffer for reading
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
            
            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                print("❌ Failed to get pixel buffer base address")
                return nil
            }
            
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            
            // Convert pixel data to MLMultiArray
            for y in 0..<height {
                for x in 0..<width {
                    let offset = y * bytesPerRow + x * 4
                    
                    // Get ARGB values (32ARGB format)
                    _ = Float(buffer[offset]) / 255.0
                    let r = Float(buffer[offset + 1]) / 255.0
                    let g = Float(buffer[offset + 2]) / 255.0
                    let b = Float(buffer[offset + 3]) / 255.0
                    
                    // Fill array based on shape requirements
                    let index = y * width + x
                    if index < mlArray.count {
                        mlArray[index] = NSNumber(value: (r + g + b) / 3.0) // Convert to grayscale
                    }
                }
            }
            
            return mlArray
            
        } catch {
            print("❌ Failed to create MLMultiArray: \(error)")
            return nil
        }
    }
} 
