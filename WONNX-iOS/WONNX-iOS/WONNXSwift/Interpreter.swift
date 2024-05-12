//
//  Interpreter.swift
//  WONNX-iOS
//
//  Created by kikemori on 2024/05/12.
//

import AVFoundation
import Foundation

class Interpreter {
    var modelPath: String
    var InputShape: (Int, Int, Int, Int) // batch_size, channels, height, width
    var outputShape: (Int, Int, Int, Int) // batch_size, channels, height, width

    init(modelPath: String, InputShape: (Int, Int, Int, Int), outputShape: (Int, Int, Int, Int)) {
        self.modelPath = modelPath
        self.InputShape = InputShape
        self.outputShape = outputShape
    }

    func loadModel() {
        var cModelPath = modelPath.cString(using: .utf8)!
        let status = cModelPath.withUnsafeMutableBufferPointer { ptr in
            load_model(
                ptr.baseAddress!,
                UInt32(InputShape.0),
                UInt32(InputShape.1),
                UInt32(InputShape.2),
                UInt32(InputShape.3),
                UInt32(outputShape.0),
                UInt32(outputShape.1),
                UInt32(outputShape.2),
                UInt32(outputShape.3)
            )
        }
        if status < 0 {
            print("Failed to load model: \(status)")
        }
    }

    func infer(pixelBuffer: CVPixelBuffer, modelInputRange: CGRect) -> [Float] {
        let preporcess_start = Date()
        guard var input = preprocess(pixelBuffer: pixelBuffer, modelInputRange: modelInputRange)
        else {
            return []
        }
        let preprocess_end = Date()
        print("preprocess: \(preprocess_end.timeIntervalSince(preporcess_start) * 1000)")

        let infer_start = Date()
        let len = input.count
        let preds = invoke(input: &input, len: len)
        let infer_end = Date()
        print("infer: \(infer_end.timeIntervalSince(infer_start) * 1000)")
        return preds
    }

    func preprocess(pixelBuffer: CVPixelBuffer, modelInputRange: CGRect) -> [Float]? {
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32BGRA)

        let modelSize = CGSize(width: CGFloat(InputShape.3), height: CGFloat(InputShape.2))
        guard let thumbnail = pixelBuffer.resize(from: modelInputRange, to: modelSize)
        else {
            return nil
        }

        // Remove the alpha component from the image buffer to get the initialized `Data`.
        guard let rgbInput = thumbnail.rgbData()
        else {
            print("Failed to convert the image buffer to RGB data.")
            return nil
        }

        // convert to channel first format
        return toChannelFirst(rgbInput)
    }

    func toChannelFirst(_ input: [Float]) -> [Float] {
        let channels = InputShape.1
        let height = InputShape.2
        let width = InputShape.3
        let batchSize = InputShape.0

        var output = [Float](repeating: 0, count: input.count)
        for b in 0..<batchSize {
            for c in 0..<channels {
                for h in 0..<height {
                    for w in 0..<width {
                        let index = b * channels * height * width + c * height * width + h * width + w
                        let newIndex = b * channels * height * width + h * width * channels + w * channels + c
                        output[newIndex] = input[index]
                    }
                }
            }
        }
        return output
    }

    func invoke(input: inout [Float], len: Int) -> [Float] {
        let output = input.withUnsafeMutableBufferPointer { ptr in
            predict(ptr.baseAddress!, UInt32(len))
        }

        let buffer = UnsafeBufferPointer(start: output.data, count: Int(output.size))
        let preds = [Float](buffer)
        return preds
    }
}
