
# Project: Training VAE, GAN, and WF Classifiers

This project provides scripts for training models using VAE, GAN, and WF classifiers on the Conflux dataset. I have also exported the Conda environment, so you can directly reproduce my results.

## Requirements

Make sure you have the necessary environment set up. You can recreate the environment using the exported Conda environment file.

## Contents

The code contains three main files for training different models:

- `train_vae.py` – for training a VAE-based generator
- `train_wgan.py` – for training a GAN-based generator
- `train_wf.py` – for training a Website Fingerprinting (WF) classifier

## Instructions

### 1. Train a VAE Generator

To train a generator using the VAE model, run:

```bash
python train_vae.py --data-path ./data/sam_conflux_dataset/ --mon-inst 200 --mon-class 100 --use-gpu --batch-size 128 --lr0 0.0001 --epoch 100000
```

Explanation:
- `--data-path`: Path to the dataset (in Wang's format, e.g., `0-0.cell`, `0-1.cell`, etc.)
- `--mon-inst`: Number of instances per class (200 in this case)
- `--mon-class`: Number of classes (100, but it can handle fewer)
- `--use-gpu`: Train on GPU
- `--batch-size`: Batch size (128)
- `--lr0`: Initial learning rate (0.0001)
- `--epoch`: Number of training epochs (100000)

> **Note:** There is a bug in the code if you're training on a GPU other than `gpu:0`. Please be mindful of this limitation.

### 2. Train a GAN Generator

To train a generator using the GAN model, run:

```bash
python train_wgan.py --data-path ./data/sam_conflux_dataset/ --mon-inst 200 --mon-class 96 --use-gpu --epochs 50000 --gpu 1
```

Explanation:
- `--data-path`: Path to the dataset
- `--mon-inst`: Number of instances per class
- `--mon-class`: Number of classes (96)
- `--use-gpu`: Train on GPU
- `--epochs`: Number of training epochs (50000)
- `--gpu`: Specify the GPU ID (1 for example)

This code should work on GPUs other than `gpu:0` without issues.

### 3. Train a WF Classifier

There are two main ways to train a WF classifier: without a generator and with a generator.

#### 3.1. Without a Generator

- **RF Model:**

```bash
python train_wf.py --data-path ./data/sam_conflux_dataset/sub/ --nosave --verbose --generator-type none --model rf --feat tam --seq-length 1600 --one-fold
```

- **DF Model:**

```bash
python train_wf.py --data-path ./data/sam_conflux_dataset/sub/ --nosave --verbose --generator-type none --model df --feat df --seq-length 10000 --one-fold
```

- **TikTok Model:**

```bash
python train_wf.py --data-path ./data/sam_conflux_dataset/sub/ --nosave --verbose --generator-type none --model df --feat tiktok --seq-length 10000 --one-fold
```

> **Note:** For DF and TikTok models, the `seq-length` is hardcoded to 10000 due to a constraint in the model initialization. The `--one-fold` option runs only one fold of a 10-fold cross-validation.

#### 3.2. With a Generator

To train a WF classifier using a pre-trained VAE or GAN generator, run:

```bash
python train_wf.py --data-path ./data/sam_conflux_dataset/sub/ --nosave --verbose --generator-type vae --generator-path ./checkpoints/vae_100_200_1600/epoch7399.pth --model rf --feat tam --seq-length 1600 --one-fold
```

Explanation:
- `--generator-type`: Specify the generator type (vae or gan)
- `--generator-path`: Path to the trained generator model
- `--seq-length`: Ensure the sequence length matches the generator's output

> **Note:** The `seq-length` argument is important, as it should be consistent for both the generator and the WF model. If you need to modify the `seq-length` logic, check lines 57-94 in `attacks/exp_rf_recover.py`. Ideally, there should be two separate `seq-length` arguments, but I didn’t have time to implement this properly.

## Troubleshooting

- **Training on GPUs other than `gpu:0`:** Be cautious when training the VAE model on GPUs other than `gpu:0` due to a known bug.
- **Hardcoded `seq-length`:** The `seq-length` is hardcoded for some models like DF and TikTok. Adjustments might be needed depending on your use case.

Feel free to contact me if you have any further questions or run into issues!
