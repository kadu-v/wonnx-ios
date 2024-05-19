use std::{collections::HashMap, sync::Arc};

use futures::executor::block_on;
use wonnx::Session;

pub struct Interpreter {
    model_path: String,
    input_shape: (usize, usize, usize, usize), // (batch_size, channels, height, width)
    output_shape: (usize, usize, usize, usize), // (batch_size, channels, height, width)
    interp: Option<Session>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            model_path: "".to_string(),
            input_shape: (0, 0, 0, 0),
            output_shape: (0, 0, 0, 0),
            interp: None,
        }
    }

    pub fn load(
        &mut self,
        model_path: &str,
        input_shape: (usize, usize, usize, usize),
        output_shape: (usize, usize, usize, usize),
    ) -> Result<(), i32> {
        println!("Loading model: {}", model_path);
        let interp = {
            let session = Session::from_path(model_path);
            block_on(session)
        };

        let interp = match interp {
            Ok(interp) => interp,
            Err(e) => {
                println!("Error: Failed to load interpreter: {:?}", e);
                return Err(-1);
            }
        };

        self.model_path = model_path.to_string();
        self.input_shape = input_shape;
        self.output_shape = output_shape;
        self.interp = Some(interp);

        println!("Model loaded successfully");
        Ok(())
    }

    /*-------------------------------------------------------------------------
    Convert channel last to channel first
    ---------------------------------------------------------------------------*/
    pub fn convert_to_channel_first(
        &self,
        input: &[f32],
    ) -> Result<Vec<f32>, i32> {
        let (_, channels, height, width) = self.input_shape;
        let mut converted_input = vec![0.0; channels * height * width];
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let idx = c * height * width + h * width + w;
                    converted_input[idx] =
                        input[h * width * channels + w * channels + c];
                }
            }
        }
        Ok(converted_input)
    }

    /*-------------------------------------------------------------------------
    Run inference
    ---------------------------------------------------------------------------*/
    pub fn predict(
        &self,
        input: &[f32],
    ) -> Result<(Vec<f32>, f32, f32, f32), i32> {
        let Some(interp) = self.interp.as_ref() else {
            println!("Error: Interpreter is not loaded");
            return Err(-2);
        };

        let preprocess_start = std::time::Instant::now();
        let input = self.convert_to_channel_first(input)?;
        let mut input_data = HashMap::new();
        input_data.insert("images".to_string(), input.as_slice().into());
        let preprocess_time = preprocess_start.elapsed().as_micros() as f32;

        // measure inference time
        let start = std::time::Instant::now();
        let output = match block_on(interp.run(&input_data)) {
            Ok(output) => output,
            Err(e) => {
                println!("Error: Failed to run interpreter: {:?}", e);
                return Err(-3);
            }
        };
        let inference_time = start.elapsed().as_millis() as f32;

        let Some(result) = output.get("output") else {
            println!("Error: Failed to get output");
            return Err(-4);
        };

        let Ok(preds) = result.try_into() else {
            println!("Error: Failed to convert output to Vec<f32>");
            return Err(-5);
        };
        let preds: &[f32] = preds;

        // Post-process output data
        let post_process_start = std::time::Instant::now();
        let preds = self.post_process(preds)?;
        let post_process_time = post_process_start.elapsed().as_micros() as f32;
        Ok((preds, preprocess_time, inference_time, post_process_time))
    }

    /*-------------------------------------------------------------------------
    Post-process output data for yolox model
    ---------------------------------------------------------------------------*/
    pub fn post_process(&self, preds: &[f32]) -> Result<Vec<f32>, i32> {
        let (_, _, height, width) = self.input_shape;
        let (_, y, x, _) = self.output_shape;

        // get classes, positions, and objectness
        let mut positions = vec![];
        let mut objectness = vec![];
        for i in 0..y {
            let offset = i * x;
            let x1 = preds[offset];
            let y1 = preds[offset + 1];
            let x2 = preds[offset + 2];
            let y2 = preds[offset + 3];
            let obj = preds[offset + 4];

            positions.push((x1, y1, x2, y2));
            objectness.push(obj);
        }

        // calculate locations
        let locs = Self::calc_locations(&positions);

        // non-max suppression
        let nms = Self::non_max_suppression(&locs, &objectness, 0.3, 0.45);

        // get final predictions
        let mut result = vec![0.; nms.len() * 6];
        for (i, (idx, bbox)) in nms.iter().enumerate() {
            // get score and classes
            let offset = idx * x;
            let (class, score) = preds[offset + 5..offset + 85]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            let (x1, y1, x2, y2) = bbox;
            result[i * 6] = class as f32;
            result[i * 6 + 1] = *score;
            result[i * 6 + 2] = *x1 / width as f32;
            result[i * 6 + 3] = *y1 / height as f32;
            result[i * 6 + 4] = *x2 / width as f32;
            result[i * 6 + 5] = *y2 / height as f32;
        }
        Ok(result)
    }

    fn calc_locations(
        positions: &[(f32, f32, f32, f32)],
    ) -> Vec<(f32, f32, f32, f32)> {
        let mut locs = vec![];

        // calc girds
        let (h, w) = (416, 416);
        let strides = vec![8, 16, 32];
        let mut h_grids = vec![];
        let mut w_grids = vec![];

        for stride in strides.iter() {
            let mut h_grid = vec![0.0; h / stride];
            let mut w_grid = vec![0.0; w / stride];

            for i in 0..h / stride {
                h_grid[i] = i as f32;
            }
            for i in 0..w / stride {
                w_grid[i] = i as f32;
            }
            h_grids.push(h_grid);
            w_grids.push(w_grid);
        }
        let acc =
            vec![0, 52 * 52, 52 * 52 + 26 * 26, 52 * 52 + 26 * 26 + 13 * 13];

        for (i, stride) in strides.iter().enumerate() {
            let h_grid = &h_grids[i];
            let w_grid = &w_grids[i];
            let idx = acc[i];

            for (i, y) in h_grid.iter().enumerate() {
                for (j, x) in w_grid.iter().enumerate() {
                    let p = idx + i * w / stride + j;
                    let (px, py, pw, ph) = positions[p];
                    let (x, y) =
                        ((x + px) * *stride as f32, (y + py) * *stride as f32);
                    let (ww, hh) =
                        (pw.exp() * *stride as f32, ph.exp() * *stride as f32);
                    let loc = (
                        x - ww / 2.0,
                        y - hh / 2.0,
                        x + ww / 2.0,
                        y + hh / 2.0,
                    );
                    locs.push(loc);
                }
            }
        }
        locs
    }

    pub fn non_max_suppression(
        boxes: &[(f32, f32, f32, f32)],
        scores: &[f32],
        score_threshold: f32,
        iou_threshold: f32,
    ) -> Vec<(usize, (f32, f32, f32, f32))> {
        let mut new_boxes = vec![];
        let mut sorted_indices = (0..boxes.len()).collect::<Vec<_>>();
        sorted_indices
            .sort_by(|a, b| scores[*a].partial_cmp(&scores[*b]).unwrap());

        while let Some(last) = sorted_indices.pop() {
            let mut remove_list = vec![];
            let score = scores[last];
            let bbox = boxes[last];
            let mut numerator = (
                bbox.0 * score,
                bbox.1 * score,
                bbox.2 * score,
                bbox.3 * score,
            );
            let mut denominator = score;

            for i in 0..sorted_indices.len() {
                let idx = sorted_indices[i];
                let (x1, y1, x2, y2) = boxes[idx];
                let (x1_, y1_, x2_, y2_) = boxes[last];
                let box1_area = (x2 - x1) * (y2 - y1);

                let inter_x1 = x1.max(x1_);
                let inter_y1 = y1.max(y1_);
                let inter_x2 = x2.min(x2_);
                let inter_y2 = y2.min(y2_);
                let inter_w = (inter_x2 - inter_x1).max(0.0);
                let inter_h = (inter_y2 - inter_y1).max(0.0);
                let inter_area = inter_w * inter_h;
                let area1 = (x2 - x1) * (y2 - y1);
                let area2 = (x2_ - x1_) * (y2_ - y1_);
                let union_area = area1 + area2 - inter_area;
                let iou = inter_area / union_area;

                if scores[idx] < score_threshold {
                    remove_list.push(i);
                } else if iou > iou_threshold {
                    remove_list.push(i);
                    let w = scores[idx] * iou;
                    numerator = (
                        numerator.0 + boxes[idx].0 * w,
                        numerator.1 + boxes[idx].1 * w,
                        numerator.2 + boxes[idx].2 * w,
                        numerator.3 + boxes[idx].3 * w,
                    );
                    denominator += w;
                } else if inter_area / box1_area > 0.7 {
                    remove_list.push(i);
                }
            }
            for i in remove_list.iter().rev() {
                sorted_indices.remove(*i);
            }
            let new_bbox = (
                numerator.0 / denominator,
                numerator.1 / denominator,
                numerator.2 / denominator,
                numerator.3 / denominator,
            );
            new_boxes.push((last, new_bbox));
        }
        new_boxes
    }
}
