// make a program to train an AI on the colour purple

// input layer = 3 neurons (RGB)
// 1 hidden layer of 4 neurons perhaps
// output = 1 neuron (Yay or Nay)

// epoch means to train the nn with all the training data once

// the learning rate decides how much to change the model in response to the estimated error

// MLP is made of many layers

use candle_core::{DType, Result, Tensor, D, Device};
use candle_nn::{loss, ops, Linear, Module, VarBuilder, VarMap};

const VOTE_DIM: usize = 2;
const RESULTS: usize = 1;
const EPOCHS: usize = 20;
const LAYER1_OUT_SIZE: usize = 4;
const LAYER2_OUT_SIZE: usize = 1;
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

    Ok(())
}