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
        let interp = {
            let session = Session::from_path(model_path);
            block_on(session)
        };

        let Ok(interp) = interp else {
            return Err(-1);
        };

        self.model_path = model_path.to_string();
        self.input_shape = input_shape;
        self.output_shape = output_shape;
        self.interp = Some(interp);

        Ok(())
    }

    pub(crate) fn predict(&self, input: &[f32]) -> Result<Vec<f32>, i32> {
        let interp = self.interp.as_ref().unwrap();
        let input = input.to_vec();
        let mut input_data = HashMap::new();
        input_data.insert("input".to_string(), input.as_slice().into());

        let Ok(output) = block_on(interp.run(&input_data)) else {
            return Err(-2);
        };

        let Some(result) = output.get("output") else {
            return Err(-3);
        };

        let Ok(preds) = result.try_into() else {
            return Err(-4);
        };

        let preds = self.post_process(preds)?;

        Ok(preds)
    }

    pub(crate) fn post_process(&self, preds: &[f32]) -> Result<Vec<f32>, i32> {
        let (_, channels, height, width) = self.output_shape;
        Ok(preds.to_vec())
    }
}
