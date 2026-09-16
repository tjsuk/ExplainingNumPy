# Extending Step 11a: Rerunning the Capstone on Full-Resolution MNIST

## What you're extending

[Step 11a](../numpy_fundamentals.ipynb) loads scikit-learn's bundled `load_digits()` dataset:
1,797 handwritten digit images, each an 8x8 grid (64 pixels), deliberately small and
low-resolution so every cell in the main notebook runs almost instantly. The famous **MNIST**
dataset is the same idea at real scale: 70,000 images, each 28x28 pixels (784 features) — the
dataset most people actually mean when they say "the digits dataset" in a machine learning
context.

## Why this matters

Every array shape in Steps 11a-11f was sized around `n_features = 64`. Rerunning the exact same
from-scratch NumPy code on data with **12.25x more pixels per image** (`784 / 64`) and **~39x
more images** (`70,000 / 1,797`) is one of the best ways to feel — not just read about — why the
speedups measured in Steps 4 and 10 matter in practice, and to see this notebook's core claim
tested for real: that the same vectorised NumPy operations scale up to genuinely large data
without changing a single line of the forward pass, loss, or gradient code.

## How to do it

### 1. Fetch the full dataset

```python
from sklearn.datasets import fetch_openml
import time

start = time.time()
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
print(f"Download/load took {time.time() - start:.1f}s")

X_mnist, y_mnist = mnist.data, mnist.target.astype(int)
print("X_mnist shape:", X_mnist.shape)   # (70000, 784)
print("y_mnist shape:", y_mnist.shape)   # (70000,)
```

The first call downloads roughly 50MB from OpenML and caches it locally — later calls load
instantly from the cache. This needs internet access the first time only.

### 2. Use MNIST's conventional train/test split

Step 11a shuffles the data with `rng.permutation` and takes the first 1,400 images for training.
MNIST instead has a **standard, universally-used split** — the first 60,000 images for training,
the last 10,000 for testing — which is what makes results comparable to published MNIST
benchmarks:

```python
x_train_raw, x_test_raw = X_mnist[:60000], X_mnist[60000:]
y_train, y_test = y_mnist[:60000], y_mnist[60000:]

print("Training images:", x_train_raw.shape[0])
print("Test images:    ", x_test_raw.shape[0])
```

`fetch_openml` already returns each image flattened to a 784-length row (unlike `load_digits()`,
whose `.images` attribute is 8x8), so there's no `.reshape(-1, n_features)` step needed here —
just the normalisation below.

### 3. Normalise pixel values — note the different scale

MNIST pixel values range **0-255** (standard 8-bit grayscale), not the 0-16 range Step 11a
divides by. Adjust the normalisation accordingly:

```python
n_features = 28 * 28   # = 784

x_train = x_train_raw / 255.0
x_test = x_test_raw / 255.0

print("Flattened training images shape:", x_train.shape)   # (60000, 784)
```

### 4. Resize the network's weights to match, and reuse everything else unchanged

This is the point of the exercise: `one_hot`, `forward_pass`, `softmax`, `cross_entropy_loss`,
and `compute_gradients` from Step 11 don't need to change **at all** — they were written in
terms of `n_features` and "however many samples are in this batch," never a hard-coded number.
Only the weight matrix's shape changes, because it's sized from `n_features`:

```python
y_train_onehot = one_hot(y_train)   # exact same function from Step 11a

rng = np.random.default_rng(seed=2)
W = rng.normal(loc=0.0, scale=0.01, size=(n_features, 10))   # now (784, 10), was (64, 10)
b = np.zeros(10)
```

### 5. Retrain — but time a few steps first

Each training step now does `(60000, 784) @ (784, 10)` instead of `(1400, 64) @ (64, 10)` — far
more arithmetic per step. Time a handful of steps before committing to Step 11e's full
300-step run, so you have a real estimate rather than a guess:

```python
start = time.time()
probs = forward_pass(x_train[:5000], W, b)
loss = cross_entropy_loss(probs, y_train_onehot[:5000])
grad_W, grad_b = compute_gradients(x_train[:5000], probs, y_train_onehot[:5000])
step_time = time.time() - start
print(f"One step on 5,000 images took {step_time:.3f}s")
print(f"Estimated time for 300 steps on all 60,000 images: {step_time * (60000/5000) * 300:.0f}s")
```

Then run the full training loop, identical in structure to Step 11e:

```python
learning_rate = 0.5
n_steps = 300
loss_history = []

start = time.time()
for step in range(n_steps):
    probs = forward_pass(x_train, W, b)
    loss = cross_entropy_loss(probs, y_train_onehot)
    grad_W, grad_b = compute_gradients(x_train, probs, y_train_onehot)

    W -= learning_rate * grad_W
    b -= learning_rate * grad_b

    loss_history.append(loss)
    if step % 50 == 0 or step == n_steps - 1:
        print(f"Step {step:3d}:  loss = {loss:.4f}")

print(f"\nTrained {n_steps} steps over {x_train.shape[0]} images in {time.time() - start:.1f} seconds")

test_probs = forward_pass(x_test, W, b)
test_predictions = np.argmax(test_probs, axis=1)
print(f"Test accuracy: {np.mean(test_predictions == y_test):.4f}")
```

## What to observe / think about

- **Compare total training time against Step 11e's original run** (300 steps over 1,400 images
  of 64 features). Work out the actual slowdown factor and compare it to the raw increase in
  arithmetic (`60000/1400 x 784/64 ≈ 524x` more multiply-adds per step) — vectorisation means
  wall-clock time won't scale anywhere near that badly, which is itself the lesson.
- **Compare final test accuracy.** Full MNIST is generally *easier* to reach high accuracy on
  than the smaller dataset, mostly because there's simply far more training data — a good
  concrete example of "more data often beats a fancier model."
- **This is where full-batch gradient descent starts to hurt.** Every one of the 300 steps above
  processes all 60,000 images before making a single weight update. Pair this guide with
  [`04_mini_batch_training.md`](04_mini_batch_training.md) to update the weights many times per
  pass through the data instead, and compare how quickly the loss curve drops per second of
  wall-clock time (not just per step).
- If you also tried [`01_hidden_layer_relu.md`](01_hidden_layer_relu.md), rerun that two-layer
  version here too — full MNIST is exactly the kind of larger, more varied dataset where an
  extra hidden layer's added capacity is more likely to show a real accuracy improvement than it
  did on the smaller 8x8 digits.

## Related NumPy tools worth knowing about

- **`np.float32` vs the default `np.float64`**: casting `x_train`/`W` to `np.float32` roughly
  halves memory use and can noticeably speed up matrix multiplication at this larger scale, at
  the cost of a little numerical precision — try `x_train.astype(np.float32)` and compare timing.
- **`np.save` / `np.load`**: save the downloaded, normalised arrays to a local `.npy` file
  (`np.save("mnist_train.npy", x_train)`) so future runs skip `fetch_openml` entirely.
