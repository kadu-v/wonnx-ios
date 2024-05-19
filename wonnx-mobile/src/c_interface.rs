use crate::interpreter::Interpreter;
use once_cell::sync::Lazy;
use std::{
    ffi::{c_char, CStr},
    sync::Mutex,
};

static INTERPRETER: Lazy<Mutex<Interpreter>> =
    Lazy::new(|| Mutex::new(Interpreter::new()));

#[repr(C)]
pub struct Array {
    pub data: *mut f32,
    pub len: i32,
    pub preprocess_time: f32,
    pub inference_time: f32,
    pub post_process_time: f32,
}

#[no_mangle]
pub extern "C" fn load_model(
    model_path: *const c_char,
    input_batch_size: usize,
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    output_batch_size: usize,
    output_channels: usize,
    output_height: usize,
    output_width: usize,
) -> i32 {
    let c_model_path = unsafe { CStr::from_ptr(model_path) };
    let Ok(model_path) = c_model_path.to_str() else {
        return -10;
    };

    let input_shape =
        (input_batch_size, input_channels, input_height, input_width);
    let output_shape = (
        output_batch_size,
        output_channels,
        output_height,
        output_width,
    );
    let mut interp = INTERPRETER.lock().unwrap();
    let _ = match interp.load(model_path, input_shape, output_shape) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("Error: {:?}", e);
            return -11;
        }
    };
    return 0;
}

#[no_mangle]
pub extern "C" fn predict(data: *mut f32, len: usize) -> Array {
    let input = unsafe { std::slice::from_raw_parts(data, len) };
    let interp = INTERPRETER.lock().unwrap();
    let Ok((mut output, preprocess_time, inference_time, post_process_time)) =
        interp.predict(input)
    else {
        return Array {
            data: std::ptr::null_mut(),
            len: 0,
            preprocess_time: 0.0,
            inference_time: 0.0,
            post_process_time: 0.0,
        };
    };
    let len = output.len();
    let data = output.as_mut_ptr();
    std::mem::forget(output); // unsafe: move ownership to caller

    Array {
        data,
        len: len as i32,
        preprocess_time,
        inference_time,
        post_process_time,
    }
}
