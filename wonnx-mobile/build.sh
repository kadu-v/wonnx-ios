#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

# Build for iOS
cargo build --target aarch64-apple-ios --release

# Build Static Library
cp target/aarch64-apple-ios/release/libWONNXMobile.a WONNXMobile/libWONNXMobile.a
cp src/c_interface.h WONNXMobile/c_interface.h