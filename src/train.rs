use candle_core::{Device, DType, D};
use candle_nn::{VarMap, VarBuilder, Optimizer, ops, loss};

use crate::{Dataset, MultiLevelPerceptron, LEARNING_RATE, EPOCHS};

pub fn train(m: Dataset, dev: &Device) -> anyhow::Result<MultiLevelPerceptron> {
    let train_labels = m.train_labels.to_device(dev)?;
    let train_images = m.train_images.to_device(dev)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = MultiLevelPerceptron::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    let test_images = m.test_images.to_device(dev)?;
    let test_labels = m.test_labels.to_device(dev)?;
    let mut final_accuracy: f32 = 0.0;
    for epoch in 1..EPOCHS + 1 {
        let logits = model.forward(&train_images)?;

        println!("{logits}, here jakey");

        // D::Minus1 might mess me up here
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;

        // check this for a vector answer rather than a scalar / 1d
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        final_accuracy = 100. * test_accuracy;
        println!("Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
                loss.to_scalar::<f32>()?,
                final_accuracy
        );
        if final_accuracy == 100.0 {
            break;
        }
    }
    println!("{final_accuracy}");

    Ok(model)

}