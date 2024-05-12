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
        intepreter = Interpreter(modelPath: modelPath, InputShape: inputShape, outputShape: outputShape)
        intepreter.loadModel()
        super.init()
    }

    override func viewDidLoad() {
        super.viewDidLoad()
    }

    // MARK: - 認識するレイヤーの初期化

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

    // MARK: - バウンディングボックスの描画

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
            width: (bbox.maxX - bbox.minX) * width,
            height: (bbox.maxY - bbox.minY) * height)
        boxLayer.frame = bounds

        // バウンディングボックスに緑色の外枠を設定
        let borderWidth = 1.0
        boxLayer.borderWidth = borderWidth
        boxLayer.borderColor = boxColor

        // 認識結果のテキストを設定
        let textLayer = CATextLayer()
        textLayer.string = text
        textLayer.fontSize = 15
        textLayer.frame = CGRect(
            x: 0, y: -20,
            width: boxLayer.frame.width,
            height: 20)
        textLayer.backgroundColor = boxColor
        textLayer.foregroundColor = textColor

        boxLayer.addSublayer(textLayer)
        detectionLayer.addSublayer(boxLayer)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let start = Date()
        let modelInputRange = detectionLayer.frame.applying(
            previewLayer.bounds.size.transformKeepAspect(toFitIn: CGSize(width: 1080, height: 1980)))
        let preds = intepreter.infer(pixelBuffer: pixelBuffer, modelInputRange: modelInputRange)
        let end = Date()
        let intervalTime = end.timeIntervalSince(start) * 1000
        print("推論時間: \(intervalTime) sec")
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
