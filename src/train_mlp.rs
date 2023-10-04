use candle_core::{Device, DType, D};
use candle_nn::{VarMap, VarBuilder, Optimizer, ops, loss};

use crate::{Dataset, MultiLevelPerceptron, LEARNING_RATE, EPOCHS};

pub fn training_mlp (
    data: Dataset,
    dev: &Device,
) -> anyhow::Result<()> {

    let train_labels = data.train_labels.to_device(&dev)?;
    let train_images = data.train_images.to_dtype(DType::U32)?.to_device(&dev)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = MultiLevelPerceptron::new(vs.clone())?;

    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    let test_images = data.test_images.to_device(&dev)?;
    let test_labels = data.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    for epoch in 1..=EPOCHS {
        let logits = model.forward(&train_images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }

    Ok(())
}