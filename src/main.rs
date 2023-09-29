// make a program to train an AI on the colour purple

// input layer = 3 neurons (RGB)
// 1 hidden layer of 4 neurons perhaps
// output = 1 neuron (Yay or Nay)

// epoch means to train the nn with all the training data once

// the learning rate decides how much to change the model in response to the estimated error

// MLP is made of many layers

use candle_core::{DType, Result, Tensor, D, Device};
use candle_nn::{loss, ops, Linear, Module, VarBuilder, VarMap, Optimizer};

// giving the inputs all in one vector so input dim = 3 (r, g, b)

const VOTE_DIM: usize = 2;
const RESULTS: usize = 1;
const EPOCHS: usize = 20;
const LAYER1_OUT_SIZE: usize = 4;
const LAYER2_OUT_SIZE: usize = 2;
const LEARNING_RATE: f64 = 0.05;

#[derive(Clone)]
pub struct Dataset {
    pub train_data_input: Tensor,
    pub train_data_results: Tensor,
    pub test_data: Tensor,
    pub test_data_results: Tensor,
}

struct MultiLevelPerceptron {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

impl MultiLevelPerceptron {

    // this makes a new MLP (the neural network)
    // ln represents a linear with the weights 
    // 	A VarBuilder is used to retrieve variables used by a model. These variables can either come from a pre-trained checkpoint, e.g. using VarBuilder::from_safetensors, or initialized for training, e.g. using
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(VOTE_DIM, LAYER1_OUT_SIZE, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_OUT_SIZE, LAYER2_OUT_SIZE, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_OUT_SIZE, RESULTS + 1, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }

    // I do not understand what is going on here
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs)
    }
}

fn main() -> anyhow::Result<()> {

    let device = Device::cuda_if_available(0)?;
    println!("{:?}", device);

    // put the data in here


    Ok(())
}

fn train(m: Dataset, dev: &Device) -> anyhow::Result<MultiLevelPerceptron> {

    // puts the training data in the gpu
    let train_results = m.train_data_results.to_device(dev)?;
    let train_votes = m.train_data_input.to_device(dev)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);

    // makes the nn
    let model = MultiLevelPerceptron::new(vs.clone())?;

    // Optimizer for Stochastic Gradient Descent.
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;

    // puts the test data in
    let test_data = m.test_data.to_device(dev)?;
    let test_results = m.test_data_results.to_device(dev)?;

    let mut final_accuracy: f32 = 0.0;

    // trains it on the test data one time per epoch
    for epoch in 1..=EPOCHS {
        let logits = model.forward(&train_votes)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_results)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_data)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_results)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_results.dims1()? as f32;
        final_accuracy = 100. * test_accuracy;
        println!("Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
                 loss.to_scalar::<f32>()?,
                 final_accuracy
        );
        if final_accuracy == 100.0 {
            break;
        }
    }
    if final_accuracy < 100.0 {
        Err(anyhow::Error::msg("The model is not trained well enough."))
    } else {
        Ok(model)
    }
}