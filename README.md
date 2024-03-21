# CheXchoNet: A Chest Radiograph Dataset with Gold Standard Echocardiography Labels

This repository is meant to serve as a resource on how to get started with the CheXchoNet data released via PhysioNet.

It contains code related to the CheXchoNet [study](https://academic.oup.com/eurheartj/advance-article/doi/10.1093/eurheartj/ehad782/7617354) and supporting code to guide those interested in using the released [dataset](https://doi.org/10.13026/kp08-ws25).

## Where is the data and how do I access it?

The data is being hosted by PhysioNet at this [link](https://doi.org/10.13026/kp08-ws25).

For data access, you must accept the data use agreement. 

## Requirements

The required libraries for running models and the jupyter notebook are included in the environment.yml file. Please install as follows.

```
conda env create -f environment.yml
conda activate chexchonet
```

## Getting Started

To get started, navigate tor the `data_exploration.ipynb` notebook. This notebook should give you an understanding of what is included in the datasets and basic summary statistics.

Once you feel comfortable with the data, you can move on to building models. The code for building and testing models is under `src/`.

## Training and Evaluating Models

1. Split the metadata file into a train, eval and test set. There is a helper function called `split_train_eval_test` in the file `./src/train/helpers.py` to help with this.
2. Create your own training configuration file. This file includes settings for model training including: how to preprocess the input data, which train/eval/test files to use, which labels to train on and hyperparameters (e.g., learning rate). You can see an example under `src/train/train_configs/example_train_config_deid.yaml`.
3. Start a visdom server in the background. This will allow you to visualize the training and testing losses and other key metrics during the training process.
```
python -m visdom.server
```
3. Train a model with the given settings using the following command (script `run.py` is under `./src/train/`). This command starts a new training run using a particular model (`dense_base_with_demo`), cuda gpu index of `0`, a visdom environment name of `CHEXONET_RUN` and the given example configuration file.
```
python -m run -m dense_base_with_demo -ci 0 -vi CHEXONET_RUN -c ./train_configs/example_train_config_deid.yaml
```
4. During training, the model will show you its progress and test itself against the evaluation dataset as well. You can visualize progress on the visdom environment at `localhost:8097`. The training run will create a new directory `./best_models/CHEXONET_RUN/` which will save the lowest evaluation loss model checkpoints and test set settings to run the test set.
5. To test your trained model, you can use the following command. This command will output AUROC/AUPRC scores for the targeted binary labels and output a file with test set predictions.
```
python -m run_test_set -ckpt {PATH TO SAVED CHECKPOINT} -ts {PATH TO TEST SET SETTINGS} -fname {OUTPUT_PREDICTIONS_SUFFIX}
```

## Citation guidelines

When using this resource, please cite:
> Elias, P., & Bhave, S. (2024). CheXchoNet: A Chest Radiograph Dataset with Gold Standard Echocardiography Labels (version 1.0.0). PhysioNet. https://doi.org/10.13026/*****.

Additionally, please cite the original publication:
> Bhave S, Rodriguez V, Poterucha T, Mutasa S, Aberle D, Capaccione KM et al. Deep learning to detect left ventricular structural abnormalities in chest X-rays. European Heart Journal 2024; 10.1093/eurheartj/ehad782

Please include the standard citation for PhysioNet:
> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.