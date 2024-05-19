use ab_glyph::{FontRef, PxScale};
use image::imageops;
use image::{imageops::FilterType, ImageBuffer, Rgb};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use log::info;
use std::time::Instant;
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
use wonnx::WonnxError;
use wonnx_mobile::interpreter::*;

/*-----------------------------------------------------------------------------
 Post processing
--------------------------------------------------------------------------------*/

fn draw_bbox(
    image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    class: &str,
    score: f32,
) {
    let x1 = x1 as u32;
    let y1 = y1 as u32;
    let x2 = x2 as u32;
    let y2 = y2 as u32;
    let rect = Rect::at(x1 as i32, y1 as i32)
        .of_size((x2 - x1) as u32, (y2 - y1) as u32);
    draw_hollow_rect_mut(image, rect, Rgb([255, 0, 0]));

    let font_data = include_bytes!("../data/fonts/CascadiaCode.ttf");
    let font = FontRef::try_from_slice(font_data).expect("error");
    let height = 10.;
    let scale = PxScale {
        x: height * 2.0,
        y: height,
    };
    draw_text_mut(
        image,
        Rgb([0, 0, 0]),
        x1 as i32,
        y1 as i32,
        scale,
        &font,
        class,
    )
}

/*-----------------------------------------------------------------------------
 Pre processing
--------------------------------------------------------------------------------*/
fn padding_image(
    image: ImageBuffer<Rgb<u8>, Vec<u8>>,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = image.dimensions();
    let target_size = if width > height { width } else { height };
    let mut new_image =
        ImageBuffer::new(target_size as u32, target_size as u32);
    let x_offset = (target_size as u32 - width) / 2;
    let y_offset = (target_size as u32 - height) / 2;
    for j in 0..height {
        for i in 0..width {
            let pixel = image.get_pixel(i, j);
            new_image.put_pixel(i + x_offset, j + y_offset, *pixel);
        }
    }
    new_image
}

fn load_image() -> (Vec<f32>, ImageBuffer<Rgb<u8>, Vec<u8>>) {
    let args: Vec<String> = std::env::args().collect();
    let image_path = if args.len() == 2 {
        Path::new(&args[1]).to_path_buf()
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("./data/images")
            .join("dog.jpg")
    };

    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> =
        image::open(image_path).unwrap().to_rgb8();
    let image_buffer = padding_image(image_buffer);
    let image_buffer =
        imageops::resize(&image_buffer, 416, 416, FilterType::Nearest);

    // convert image to Vec<f32> with channel first format
    let image = image_buffer.to_vec();
    let mut converted_image = vec![0.0; image.len()];

    // convert u8 to f32
    for i in 0..image.len() {
        converted_image[i] = image[i] as f32;
    }

    (converted_image, image_buffer)
}

fn get_coco_labels() -> Vec<String> {
    // Download the ImageNet class labels, matching SqueezeNet's classes.
    let labels_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("./data/models")
        .join("coco-classes.txt");
    let file = BufReader::new(fs::File::open(labels_path).unwrap());

    file.lines().map(|line| line.unwrap()).collect()
}

/*-----------------------------------------------------------------------------
 Main
--------------------------------------------------------------------------------*/
// Hardware management
async fn execute_gpu() -> Result<Vec<f32>, WonnxError> {
    // ONNX Interpreter
    let mut interp = Interpreter::new();

    // Load Model
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("./data/models")
        .join("yolox_nano.onnx");
    let input_shape = (1, 3, 416, 416);
    let output_shape = (1, 3549, 85, 1);
    interp
        .load(model_path.to_str().unwrap(), input_shape, output_shape)
        .unwrap();

    info!("Start Compute");
    let (image, image_buffer) = load_image();
    let (preds, prerocess_time, inference_time, postprocess_time) =
        interp.predict(&image).unwrap();

    // Draw bounding box
    let coco_labels = get_coco_labels();
    let mut image_buffer = image_buffer;

    for i in 0..preds.len() / 6 {
        let offset = i * 6;
        let class = preds[offset];
        let score = preds[offset + 1];
        let x1 = preds[offset + 2] * input_shape.2 as f32;
        let y1 = preds[offset + 3] * input_shape.3 as f32;
        let x2 = preds[offset + 4] * input_shape.2 as f32;
        let y2 = preds[offset + 5] * input_shape.3 as f32;
        draw_bbox(
            &mut image_buffer,
            x1 as f32,
            y1 as f32,
            x2 as f32,
            y2 as f32,
            coco_labels[class as usize].as_str(),
            score,
        );
        println!(
            "class: {}, score: {}, x0: {}, y0: {}, x1: {}, y1: {}",
            coco_labels[class as usize], score, x1, y1, x2, y2
        );
    }
    image_buffer.save("./data/images/output_dog.jpg").unwrap();

    println!(
        "preprocess_time: {:#?} ms, inference_time: {:#?} ms, postprocess_time: {:#?} ms",
        prerocess_time, inference_time, postprocess_time
    );
    Ok(preds)
}

async fn run() {
    // Output shape is [1, 3549, 85]
    // 85 = 4 (bounding box) + 1 (objectness) + 80 (class probabilities)
    let outputs = execute_gpu().await.unwrap();
}

fn main() {
    env_logger::init();
    let time_pre_compute = Instant::now();

    pollster::block_on(run());
    let time_post_compute = Instant::now();
    println!("time: main: {:#?}", time_post_compute - time_pre_compute);
}
