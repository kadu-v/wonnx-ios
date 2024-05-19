#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

# Build for iOS
export WGPU_BACKEND=metal
export WGPU_POWER_PREF=high 
cargo build --target aarch64-apple-ios --release

# Copy Static Library
cp target/aarch64-apple-ios/release/libwonnx_mobile.a WONNXMobile/libwonnx_mobile.a
cp src/c_interface.h WONNXMobile/c_interface.h

# Copy Headers and Binary to iOS Project
cp -r WONNXMobile/* ../ios/WONNX-iOS/WONNX-iOS