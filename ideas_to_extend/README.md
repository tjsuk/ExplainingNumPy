# Ideas to Extend — Detailed Guides

Each file here expands one item from `numpy_fundamentals.ipynb`'s closing "Ideas to extend"
section into a full, self-contained walkthrough: why the idea matters, a step-by-step
explanation of how to do it, complete runnable code, and things to try afterward to deepen the
understanding. They assume you've already worked through the main notebook, and reference its
variables (`x_train`, `W`, `b`, `forward_pass`, `compute_gradients`, etc.) directly.

| Guide | Extends | What you'll learn |
|---|---|---|
| [01_hidden_layer_relu.md](01_hidden_layer_relu.md) | Step 11 | Adding a ReLU hidden layer and deriving its gradient with the chain rule — turning softmax regression into a true multi-layer network |
| [02_full_mnist.md](02_full_mnist.md) | Step 11a | Rerunning the capstone at ~39x the data scale on full-resolution MNIST, reusing every function unchanged |
| [03_learning_rate_experiments.md](03_learning_rate_experiments.md) | Step 11e | Seeing both learning-rate failure modes (too slow, unstable/diverging) directly, instead of just reading about them |
| [04_mini_batch_training.md](04_mini_batch_training.md) | Step 11e | Splitting training into shuffled mini-batches instead of one full-batch update per step, and weighing the noise/speed trade-off |
| [05_compare_tensorflow_pytorch.md](05_compare_tensorflow_pytorch.md) | Step 11 | Rebuilding the exact same model in TensorFlow/Keras and PyTorch, and comparing autodiff against the `compute_gradients` you derived by hand |

## Suggested order

If you want to work through all five, this order builds on itself reasonably well:

1. **03** (learning rate) — quick, no new dependencies, just rerun Step 11e differently
2. **04** (mini-batch training) — extends the same training loop you already know
3. **01** (hidden layer + ReLU) — deepens the model itself, still pure NumPy
4. **02** (full MNIST) — same model as Step 11, just at real scale
5. **05** (compare with TensorFlow/PyTorch) — introduces new frameworks, so do it last
