use std::collections::HashMap;

use futures::executor::block_on;
use wonnx::Session;

pub(crate) struct Interpreter {
    model_path: String,
    input_shape: (usize, usize, usize, usize), // (batch_size, channels, height, width)
    output_shape: (usize, usize, usize, usize), // (batch_size, channels, height, width)
    interp: Option<Session>,
}

impl Interpreter {
    pub(crate) fn new() -> Self {
        Self {
            model_path: "".to_string(),
            input_shape: (0, 0, 0, 0),
            output_shape: (0, 0, 0, 0),
            interp: None,
        }
    }

    pub(crate) fn load(
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

        let Ok(interp) = interp else {
            println!("Error: Failed to load interpreter");
            return Err(-1);
        };

        self.model_path = model_path.to_string();
        self.input_shape = input_shape;
        self.output_shape = output_shape;
        self.interp = Some(interp);

        println!("Model loaded successfully");
        Ok(())
    }

    pub(crate) fn predict(&self, input: &[f32]) -> Result<Vec<f32>, i32> {
        let Some(interp) = self.interp.as_ref() else {
            println!("Error: Interpreter is not loaded");
            return Err(-2);
        };

        let input = input.to_vec();
        let mut input_data = HashMap::new();
        input_data.insert("images".to_string(), input.as_slice().into());

        let output = match block_on(interp.run(&input_data)) {
            Ok(output) => output,
            Err(e) => {
                println!("Error: Failed to run interpreter: {:?}", e);
                return Err(-3);
            }
        };

        let Some(result) = output.get("output") else {
            println!("Error: Failed to get output");
            return Err(-4);
        };

        let Ok(preds) = result.try_into() else {
            println!("Error: Failed to convert output to Vec<f32>");
            return Err(-5);
        };

        let preds = self.post_process(preds)?;

        Ok(preds)
    }

    pub(crate) fn post_process(&self, preds: &[f32]) -> Result<Vec<f32>, i32> {
        let (_, channels, height, width) = self.output_shape;
        Ok(preds.to_vec())
    }
}
