import Accelerate
import CoreML

class AudioProcessor {
    // Constants for mel spectrogram generation
    private let sampleRate: Double = 16000
    private let hopLength: Int = 200    // For 25fps video
    private let nFft: Int = 800
    private let nMels: Int = 80
    private let fMin: Double = 0.0
    private let fMax: Double = 8000.0
    
    // Pre-compute window and filterbank
    private let hanningWindow: [Float]
    private var melFilterbank: MLMultiArray?
    
    init() {
        // Create Hanning window at initialization
        hanningWindow = vDSP.window(ofType: Float.self,
                                  usingSequence: .hanningDenormalized,
                                  count: nFft,
                                  isHalfWindow: false)
        
        setupMelFilterbank()
        print("🎵 AudioProcessor initialized:")
        print("- Sample rate: \(sampleRate) Hz")
        print("- Hop length: \(hopLength)")
        print("- FFT size: \(nFft)")
        print("- Mel bands: \(nMels)")
    }
    
    private func setupMelFilterbank() {
        let melMin = frequencyToMel(fMin)
        let melMax = frequencyToMel(fMax)
        let melPoints = Array(stride(from: melMin, 
                                   through: melMax, 
                                   by: (melMax - melMin) / Double(nMels + 1)))
        let freqPoints = melPoints.map { melToFrequency($0) }
        
        let fftFreqs = Array(0..<(nFft/2 + 1)).map { Double($0) * sampleRate / Double(nFft) }
        
        do {
            melFilterbank = try MLMultiArray(shape: [1, NSNumber(value: nMels), NSNumber(value: nFft/2 + 1)],
                                           dataType: .float32)
            
            for i in 0..<nMels {
                let filterRange = createTriangleFilter(freqPoints[i],
                                                     freqPoints[i + 1],
                                                     freqPoints[i + 2],
                                                     fftFreqs: fftFreqs)
                
                for (j, value) in filterRange.enumerated() {
                    melFilterbank?[[0, i as NSNumber, j as NSNumber]] = value as NSNumber
                }
            }
            
            print("✅ Mel filterbank shape: \(melFilterbank?.shape.map { $0.intValue } ?? [])")
            
        } catch {
            print("❌ Error creating mel filterbank: \(error)")
        }
    }
    
    public func audioToMelSpectrogram(audioSamples: [Float]) -> MLMultiArray? {
        print("🔄 Processing audio with \(audioSamples.count) samples")

        // ✅ Step 1: Debug Audio Input Range
        let audioMin = audioSamples.min() ?? 0
        let audioMax = audioSamples.max() ?? 0
        print("🎤 Audio Sample Range: Min \(audioMin), Max \(audioMax)")

        // Ensure valid audio range (avoid silent input)
        if audioMin == 0 && audioMax == 0 {
            print("❌ ERROR: Audio input is silent! Ensure microphone is recording.")
            return nil
        }

        // ✅ Step 2: Normalize Audio to [-1, 1] Range
        let maxAbsValue = max(abs(audioMin), abs(audioMax))
        let normalizedSamples = maxAbsValue > 0 ? audioSamples.map { $0 / maxAbsValue } : audioSamples
        print("🔊 Normalized Audio Range: Min \(normalizedSamples.min() ?? 0), Max \(normalizedSamples.max() ?? 0)")

        do {
            // ✅ Step 3: Initialize Output Array with Correct Shape [5, 1, 80, 16]
            let melSpectrogram = try MLMultiArray(shape: [5, 1, 80, 16], dataType: .float16)

            // ✅ Step 4: Compute STFT
            let stft = computeSTFT(samples: normalizedSamples)
            if stft.isEmpty {
                print("❌ ERROR: STFT computation returned empty frames.")
                return nil
            }
            print("✅ STFT computed, \(stft.count) frames generated.")

            // ✅ Step 5: Compute Power Spectrogram
            let powerSpectrogram = computePowerSpectrogram(stft: stft)
            if powerSpectrogram.isEmpty {
                print("❌ ERROR: Power spectrogram is empty!")
                return nil
            }
            print("✅ Power spectrogram computed with \(powerSpectrogram.count) frames, \(powerSpectrogram[0].count) bins")

            // ✅ Step 6: Apply Mel Filterbank and Fill Output Array
            for batch in 0..<5 {
                    for timeIdx in 0..<16 {
                        let actualIdx = min(timeIdx, powerSpectrogram.count - 1)

                        // Apply mel filterbank to single frame
                        if let melFrame = applyMelFilterbank(to: [powerSpectrogram[actualIdx]]) {
                            for melBin in 0..<80 {
                                let indices = [batch, 0, melBin, timeIdx] as [NSNumber]
                                let value = melFrame[[0, melBin as NSNumber]].floatValue
                                let logValue = log10(max(value, 1e-10)) // Apply log scaling, avoid log(0)
                                melSpectrogram[indices] = NSNumber(value: logValue)
                            }
                        } else {
                            print("❌ ERROR: Failed to apply mel filterbank for frame \(actualIdx)")
                            return nil
                        }
                    }
                }
            // ✅ Step 7: Final Verification Before Return
            let melSpectrogramArray = (0..<melSpectrogram.count).map { melSpectrogram[$0].floatValue }
            let melMin = melSpectrogramArray.min() ?? 0
            let melMax = melSpectrogramArray.max() ?? 0

            print("✅ Returning Mel Spectrogram with shape:", melSpectrogram.shape.map { $0.intValue })
            print("📊 Mel Spectrogram Value Range: Min \(melMin), Max \(melMax)")


            return melSpectrogram  // ✅ Ensure this is returned correctly

        } catch {
            print("❌ ERROR: Exception encountered while creating Mel spectrogram: \(error)")
            return nil
        }
    }

    
    private func applyMelFilterbank(to powerSpectrogram: [[Float]]) -> MLMultiArray? {
        guard let filterbank = melFilterbank else {1
            print("❌ Mel filterbank not initialized")
            return nil
        }
        
        // Verify input shapes
        let frameSize = powerSpectrogram[0].count
        let expectedSize = nFft/2 + 1
        guard frameSize == expectedSize else {
            print("❌ Power spectrogram frame size mismatch: got \(frameSize), expected \(expectedSize)")
            return nil
        }
        
        do {
            let result = try MLMultiArray(shape: [1, 80], dataType: .float32)
            
            // Apply filterbank
            for i in 0..<nMels {
                var sum: Float = 0.0
                for j in 0..<frameSize {
                    let powerValue = powerSpectrogram[0][j]
                    let filterValue = filterbank[[0, i as NSNumber, j as NSNumber]].floatValue
                    sum += powerValue * filterValue
                }
                result[[0, i as NSNumber]] = NSNumber(value: sum)
            }
            
            return result
            
        } catch {
            print("❌ Error in mel filterbank application: \(error)")
            return nil
        }
    }
    
    private func computeSTFT(samples: [Float]) -> [[Complex<Float>]] {
        var stft: [[Complex<Float>]] = []
        let windowSize = nFft
        let hopSize = hopLength
        
        // Process frames with sliding window
        var startIdx = 0
        while startIdx + windowSize <= samples.count {
            // Extract frame and apply Hanning window
            var frame = Array(samples[startIdx..<startIdx + windowSize])
            vDSP.multiply(frame, hanningWindow, result: &frame)
            
            // Compute FFT
            let fft = FFT(frame: frame)
            stft.append(fft)
            
            // Slide window
            startIdx += hopSize
        }
        
        return stft
    }
    
    private func computePowerSpectrogram(stft: [[Complex<Float>]]) -> [[Float]] {
        return stft.map { frame in
            // Only take first nFft/2 + 1 bins to match mel filterbank shape
            frame.prefix(nFft/2 + 1).map { complex in
                let real = complex.real
                let imag = complex.imaginary
                return real * real + imag * imag
            }
        }
    }
    
    // Helper functions for Mel spectrogram computation
    private func frequencyToMel(_ frequency: Double) -> Double {
        // Convert frequency to mel scale using formula:
        // mel = 2595 * log10(1 + f/700)
        return 2595 * log10(1 + frequency / 700)
    }
    
    private func melToFrequency(_ mel: Double) -> Double {
        // Convert mel to frequency using inverse formula:
        // f = 700 * (10^(mel/2595) - 1)
        return 700 * (pow(10, mel / 2595) - 1)
    }
    
    private func createTriangleFilter(_ start: Double, _ peak: Double, _ end: Double, fftFreqs: [Double]) -> [Float] {
        // Create triangular filter for mel filterbank
        return fftFreqs.map { freq -> Float in
            if freq < start || freq > end {
                // Outside filter range
                return 0
            } else if freq <= peak {
                // Left side of triangle
                return Float((freq - start) / (peak - start))
            } else {
                // Right side of triangle
                return Float((end - freq) / (end - peak))
            }
        }
    }
}

// Helper Complex number struct
struct Complex<T: FloatingPoint> {
    var real: T
    var imaginary: T
}

extension AudioProcessor {
    private func FFT(frame: [Float]) -> [Complex<Float>] {
        let log2n = vDSP_Length(log2(Double(frame.count)))
        let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
        defer { vDSP_destroy_fftsetup(setup) }
        
        var realp = [Float](frame)
        var imagp = [Float](repeating: 0.0, count: frame.count)
        
        // Create complex split array using a different approach
        var complex = DSPSplitComplex(realp: realp.withUnsafeMutableBufferPointer { $0.baseAddress! },
                                     imagp: imagp.withUnsafeMutableBufferPointer { $0.baseAddress! })
        
        // Forward FFT
        vDSP_fft_zip(setup, &complex, 1, log2n, FFTDirection(FFT_FORWARD))
        
        // Convert to array of Complex numbers
        var result = [Complex<Float>]()
        for i in 0..<frame.count {
            result.append(Complex(real: realp[i], imaginary: imagp[i]))
        }
        
        return result
    }
}

// Helper extension to safely calculate flat index
extension MLMultiArray {
    func indexIntoArray(indices: [NSNumber]) -> Int {
        var index = 0
        var stride = 1
        
        for i in (0..<indices.count).reversed() {
            index += indices[i].intValue * stride
            if i > 0 {
                stride *= shape[i].intValue
            }
        }
        
        return index
    }
} 
