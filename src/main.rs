mod train;
mod train_mlp;

use candle_core::{Device, Result, Tensor, DType, D};
use candle_nn::{Linear, Module, VarBuilder, VarMap, ops, loss};
use train::train;
use rust_mnist::Mnist;
use train_mlp::training_mlp;

const LEARNING_RATE: f64 = 0.05;
const EPOCHS: usize = 100000;

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
    let device = Device::cuda_if_available(0)?;
    let mnist = Mnist::new("MNIST_data/");
    
    // let train_image = mnist.train_data.clone()[0].map(|a| a as f32);
    // let train_label = vec![mnist.train_labels[0]];

    // let train_img_tensor: Tensor = Tensor::from_vec(train_image.to_vec(), (1, 784), &device)?;
    // let train_label_tensor: Tensor = Tensor::from_vec(train_label, 1, &device)?.to_dtype(DType::U32)?;

    // let varmap = VarMap::new();
    // let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    // let model = MultiLevelPerceptron::new(vs.clone())?;

    // let a = model.forward(&train_img_tensor)?;

    // println!("a: {a}\n\n");

    // let soft_max = ops::log_softmax(&a, D::Minus1)?;
    // println!("b: {soft_max}\n\n");

    // let loss = loss::nll(&soft_max, &train_label_tensor)?;
    // println!("c: {loss}\n\n");

    let train_images: Vec<f32> = mnist.train_data.clone().iter().flat_map(|a| a.iter().map(|a| a.clone() as f32)).collect();
    let train_labels: Vec<u8> = mnist.train_labels.clone();
    let num_images = mnist.train_data.len();
    let test_images: Vec<f32> = mnist.test_data.clone().iter().flat_map(|a| a.iter().map(|a| a.clone() as f32)).collect();
    let test_labels: Vec<u8> = mnist.test_labels.clone();
    let num_test_images = mnist.test_data.len();

    let trn_imgs_tensor: Tensor = Tensor::from_vec(train_images, (num_images, 784), &device)?;
    let trn_lbls_tensor: Tensor = Tensor::from_vec(train_labels, num_images, &device)?;
    let tst_imgs_tensor: Tensor = Tensor::from_vec(test_images, (num_test_images, 784), &device)?;
    let tst_lbls_tensor: Tensor = Tensor::from_vec(test_labels, num_test_images, &device)?;

    let data = Dataset {
        train_images: trn_imgs_tensor,
        train_labels: trn_lbls_tensor,
        test_images: tst_imgs_tensor,
        test_labels: tst_lbls_tensor,
    };

    // train(data, &device);
    match training_mlp(data, &device) {
        Ok(_) => (),
        Err(a) => println!("{a}"),
    }

    Ok(())
}