# Data Directory: Day 28 — CS231n CNNs for Visual Recognition

This directory stores the CIFAR-10 dataset used by `train_minimal.py`.

## Dataset

**CIFAR-10** (Canadian Institute for Advanced Research, 10 classes)
- 60,000 32x32 color images in 10 classes
- 50,000 training images, 10,000 test images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Source: https://www.cs.toronto.edu/~kriz/cifar.html

## Usage

The training script downloads CIFAR-10 automatically on first run:

```bash
python train_minimal.py
```

The download creates a `cifar-10-batches-py/` subdirectory here. To use a copy elsewhere, pass `--data /path/to/cifar-10-batches-py`.

## Note on Training Speed

Our `implementation.py` is pure NumPy on the CPU: im2col convolutions and hand-written backprop. The default run trains on the first 10,000 of the 50,000 training images for 6 epochs (about 1.5 minutes on one CPU core, ~51% test accuracy); `--long` uses all 50,000 for 5 epochs (about 6.5 minutes, ~63%). That is enough to see a CNN learn, not to reach state-of-the-art accuracy; for that, use PyTorch or another framework with GPU support.
