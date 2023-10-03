mod train;

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use train::train;
use rust_mnist::Mnist;

const LEARNING_RATE: f64 = 0.05;
const EPOCHS: usize = 10;

pub struct MultiLevelPerceptron {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

impl MultiLevelPerceptron {
    fn new(vs: VarBuilder) -> Result<Self> {

        // 784 in
        let ln1 = candle_nn::linear(784, 100, vs.pp("ln1"))?;
        // 100 in
        let ln2 = candle_nn::linear(100, 32, vs.pp("ln2"))?;
        // 32 in
        let ln3 = candle_nn::linear(32, 10, vs.pp("ln3"))?;
        // 10 out
        Ok(Self { ln1, ln2, ln3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs)
    }
}

#[derive(Clone)]
pub struct Dataset {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
}

fn main() -> Result<()> {
    // Use Device::new_cuda(0)?; to use the GPU.
    let device = Device::cuda_if_available(0)?;
    let mnist = Mnist::new("MNIST_data/");
    
    let train_images = mnist.train_data.clone()[0];
    let train_labels = mnist.train_labels.clone()[0];

    println!("{}, {}", train_images.len(), train_labels);

    // train(data, &device);

    Ok(())
}