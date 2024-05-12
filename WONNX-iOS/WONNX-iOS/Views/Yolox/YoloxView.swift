//
//  YoloxView.swift
//  WONNX-iOS
//
//  Created by kikemori on 2024/05/13.
//

import Foundation
import SwiftUI

struct YoloxView: View {
    @State var aspectRatio: Float = 1.0
    var body: some View {
        HostedYoloxViewController(aspectRatio: $aspectRatio).ignoresSafeArea()
    }
}
