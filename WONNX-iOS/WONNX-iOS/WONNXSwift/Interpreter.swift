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
    var cocoLabels: [String]

    init(modelPath: String,
         InputShape: (Int, Int, Int, Int),
         outputShape: (Int, Int, Int, Int),
         labelPath: String)
    {
        self.modelPath = modelPath
        self.InputShape = InputShape
        self.outputShape = outputShape
        self.cocoLabels = try! String(contentsOfFile: labelPath).components(separatedBy: "\n")
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

    func get_class(index: Int) -> String {
        return cocoLabels[index]
    }

    func infer(pixelBuffer: CVPixelBuffer, modelInputRange: CGRect) -> ([Float], Float, Float, Float) {
        guard var input = preprocess(pixelBuffer: pixelBuffer, modelInputRange: modelInputRange)
        else {
            return ([], 0.0, 0.0, 0.0)
        }

        let len = input.count
        let preds = invoke(input: &input, len: len)
        return preds
    }

    func preprocess(pixelBuffer: CVPixelBuffer, modelInputRange: CGRect) -> [Float]? {
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32BGRA)

        // measure the input image size
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

        return rgbInput
    }

    func invoke(input: inout [Float], len: Int) -> ([Float], Float, Float, Float) {
        let output = input.withUnsafeMutableBufferPointer { ptr in
            predict(ptr.baseAddress!, UInt32(len))
        }

        let buffer = UnsafeBufferPointer(start: output.data, count: Int(output.len))
        let preds = [Float](buffer)
        let preprocessTime = output.preprocess_time
        let inferTime = output.inference_time
        let postprocessTime = output.post_process_time
        return (preds, preprocessTime, inferTime, postprocessTime)
    }
}
