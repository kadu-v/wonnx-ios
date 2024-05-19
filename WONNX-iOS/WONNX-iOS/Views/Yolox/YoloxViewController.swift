//
//  YoloxViewController.swift
//  WONNX-iOS
//
//  Created by kikemori on 2024/05/13.
//

import AVFoundation
import os
import SwiftUI
import UIKit

class YoloxViewController: ViewController {
    var detectionLayer: CALayer!
    var intepreter: Interpreter
    // モデルのアスペクト比
    @Binding var aspectRatio: Float

    init(aspectRatio: Binding<Float>) {
        _aspectRatio = aspectRatio

        // モデルの初期化
        let modelPath = Bundle.main.path(forResource: "yolox_nano", ofType: "onnx")!
        let inputShape = (1, 3, 416, 416)
        let outputShape = (1, 3549, 85, 1)
        let cocoLabelsPath = Bundle.main.path(forResource: "coco-classes", ofType: "txt")!
        intepreter = Interpreter(modelPath: modelPath,
                                 InputShape: inputShape,
                                 outputShape: outputShape,
                                 labelPath: cocoLabelsPath)
        intepreter.loadModel()
        super.init()
    }

    override func viewDidLoad() {
        super.viewDidLoad()
    }

    override func setupLayers() {
        // 検出結果を表示させるレイヤーを作成
        let width = previewBounds.width
        let height = previewBounds.width * CGFloat(aspectRatio)
        detectionLayer = CALayer()
        detectionLayer.frame = CGRect(
            x: 0,
            y: 0,
            width: width,
            height: height)
        detectionLayer.position = CGPoint(
            x: previewBounds.midX,
            y: previewBounds.midY)

        // detectionLayer に緑色の外枠を設定
        let borderWidth = 3.0
        let boxColor = UIColor.green.cgColor
        detectionLayer.borderWidth = borderWidth
        detectionLayer.borderColor = boxColor

        DispatchQueue.main.async { [weak self] in
            if let layer = self?.previewLayer {
                layer.addSublayer(self!.detectionLayer)
            }
        }
    }

    func drawTime(preprocessTime: Float, inferTime: Float, postprocessTime: Float) {
        let text = String(format: "Preprocess: %.2fus, \nInfer: %.2fms, \nPostprocess: %.2fus",
                          preprocessTime, inferTime, postprocessTime)
        let textLayer = CATextLayer()
        textLayer.string = text
        textLayer.fontSize = 15
        textLayer.frame = CGRect(
            x: 0, y: -20 * 3,
            width: detectionLayer.frame.width,
            height: 20 * 3)
        textLayer.backgroundColor = UIColor.clear.cgColor
        textLayer.foregroundColor = UIColor.green.cgColor
        detectionLayer.addSublayer(textLayer)
    }

    func drawDetection(
        bbox: CGRect,
        text: String,
        boxColor: CGColor = UIColor.green.withAlphaComponent(0.5).cgColor,
        textColor: CGColor = UIColor.black.cgColor)
    {
        let boxLayer = CALayer()

        // バウンディングボックスの座標を計算
        let width = detectionLayer.frame.width
        let height = detectionLayer.frame.width
        let bounds = CGRect(
            x: bbox.minX * width,
            y: bbox.minY * height,
            width: (bbox.maxX - bbox.minX) * width + 10,
            height: (bbox.maxY - bbox.minY) * height + 10)
        boxLayer.frame = bounds

        // バウンディングボックスに緑色の外枠を設定
        let borderWidth = 3.0
        boxLayer.borderWidth = borderWidth
        boxLayer.borderColor = boxColor

        // 認識結果のテキストを設定
        let textLayer = CATextLayer()
        textLayer.string = text
        textLayer.fontSize = 15
        textLayer.frame = CGRect(
            x: 0, y: -20 * 2,
            width: boxLayer.frame.width,
            height: 20 * 2)
        textLayer.backgroundColor = boxColor
        textLayer.foregroundColor = textColor

        boxLayer.addSublayer(textLayer)
        detectionLayer.addSublayer(boxLayer)
    }

    func drawDetections(preds: [Float],
                        preprocessTime: Float,
                        inferenceTime: Float,
                        postprocessTime: Float)
    {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        detectionLayer.sublayers = nil
        drawTime(preprocessTime: preprocessTime,
                 inferTime: inferenceTime,
                 postprocessTime: postprocessTime)
        for i in 0 ..< preds.count / 6 {
            let offset = i * 6
            let cls = preds[offset + 0]
            let score = preds[offset + 1]
            let x1 = preds[offset + 2]
            let y1 = preds[offset + 3]
            let x2 = preds[offset + 4]
            let y2 = preds[offset + 5]

            let bbox = CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(x2 - x1), height: CGFloat(y2 - y1))
            let label = intepreter.get_class(index: Int(cls))
            drawDetection(bbox: bbox, text: label + "\n" + String(format: "%.2f", score))
        }
        CATransaction.commit()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let start = Date()
        let modelInputRange = detectionLayer.frame.applying(
            previewLayer.bounds.size.transformKeepAspect(toFitIn: CGSize(width: 1080, height: 1980)))
        let (preds, preprocessTime, inferenceTime, postprocessTime)
            = intepreter.infer(pixelBuffer: pixelBuffer, modelInputRange: modelInputRange)
        let end = Date()
        let intervalTime = end.timeIntervalSince(start) * 1000
        print("推論時間: \(intervalTime) sec")

        Task { @MainActor in
            drawDetections(preds: preds,
                           preprocessTime: preprocessTime,
                           inferenceTime: inferenceTime,
                           postprocessTime: postprocessTime)
        }
    }
}

struct HostedYoloxViewController: UIViewControllerRepresentable {
    @Binding var aspectRatio: Float
    func makeUIViewController(context: Context) -> YoloxViewController {
        return YoloxViewController(aspectRatio: $aspectRatio)
    }

    func updateUIViewController(_ uiViewController: YoloxViewController, context: Context) {
        guard uiViewController.detectionLayer != nil else {
            return
        }
        uiViewController.aspectRatio = aspectRatio
        uiViewController.detectionLayer.frame = CGRect(
            x: uiViewController.detectionLayer.frame.minX,
            y: uiViewController.detectionLayer.frame.minY,
            width: uiViewController.detectionLayer.frame.width,
            height: uiViewController.detectionLayer.frame.width * CGFloat(aspectRatio))
    }
}
