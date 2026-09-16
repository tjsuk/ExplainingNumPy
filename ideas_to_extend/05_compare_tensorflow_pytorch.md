# Extending Step 11: Rebuilding It with TensorFlow/Keras or PyTorch

## What you're extending

Step 11 built softmax regression entirely from scratch: a weight matrix and bias you initialise
yourself, a hand-written `forward_pass`, a hand-written `cross_entropy_loss`, and hand-derived
gradients in `compute_gradients`, all wired together by a manual training loop in Step 11e. This
guide rebuilds the *exact same model* — one Dense layer, softmax, cross-entropy loss, gradient
descent, on the same digits dataset — using a real deep learning framework, so you can compare
what a framework does automatically against the code you already trust because Step 12 verified
it.

## Why this matters

The whole point of Step 11 was to demystify what `model.fit(...)` does inside a real framework.
Having built it by hand, the framework version should now read as *recognisable* rather than
magical: every piece — the Dense layer, the softmax activation, the cross-entropy loss, gradient
descent — has a direct one-to-one counterpart in the code you already wrote and verified. The
main difference isn't the maths; it's that the framework computes gradients **automatically**
("autodiff") instead of you deriving `compute_gradients` by hand, and can transparently run the
same vectorised operations on a GPU instead of just the CPU-bound NumPy this notebook used
throughout.

## How to do it

### Option A: TensorFlow/Keras

```python
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.images.reshape(-1, 64) / 16.0   # same flatten + normalise as Step 11a/11's x_train
y = digits.target

X_train, X_test, y_train_tf, y_test_tf = train_test_split(
    X, y, test_size=0.2, random_state=1, shuffle=True
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="softmax", input_shape=(64,)),   # exactly W, b + softmax
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),   # plain gradient descent, same as Step 11e
    loss="sparse_categorical_crossentropy",                  # same formula as Step 11c's cross_entropy_loss
    metrics=["accuracy"],
)

history = model.fit(X_train, y_train_tf, epochs=300, batch_size=len(X_train), verbose=0)
# batch_size=len(X_train) reproduces Step 11e's full-batch training exactly, rather than Keras's
# usual mini-batch default (see 04_mini_batch_training.md for what changing this does)

test_loss, test_accuracy = model.evaluate(X_test, y_test_tf, verbose=0)
print(f"Keras test accuracy: {test_accuracy:.4f}")
```

### Option B: PyTorch

```python
import torch
import torch.nn as nn

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_tf, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test_tf, dtype=torch.long)

model = nn.Linear(64, 10)   # exactly W, b - no separate softmax module needed (see note below)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
loss_fn = nn.CrossEntropyLoss()   # combines softmax + cross-entropy in one numerically-stable step

for step in range(300):
    optimizer.zero_grad()
    scores = model(X_train_t)             # X @ W.T + b - PyTorch stores weights transposed vs. Step 11
    loss = loss_fn(scores, y_train_t)     # softmax applied INSIDE this loss, not as a separate layer
    loss.backward()                        # autodiff - PyTorch works out compute_gradients for you
    optimizer.step()                       # W -= learning_rate * grad_W, done for you

    if step % 50 == 0 or step == 299:
        print(f"Step {step:3d}: loss = {loss.item():.4f}")

with torch.no_grad():
    test_predictions = model(X_test_t).argmax(dim=1)
    accuracy = (test_predictions == y_test_t).float().mean()
    print(f"PyTorch test accuracy: {accuracy:.4f}")
```

`nn.CrossEntropyLoss` deliberately does *not* take already-softmaxed probabilities — it applies
softmax internally, combined with the log and negative-mean-log steps from Step 11c, in one
numerically stable operation. This is the same reason Step 11b's `softmax` function subtracts
each row's max before exponentiating: avoiding overflow in `exp()`.

## What to observe / think about

- **Compare final test accuracy and the loss curve shape against Step 11's own results.** With
  the same effective learning rate, full-batch updates, and number of steps, they should land
  very close to each other — any remaining gap is mostly down to differences in weight
  initialisation, not the underlying maths, which is identical.
- **Print the trained Keras/PyTorch weight shapes and compare them to `W`'s `(64, 10)` shape from
  Step 11.** (PyTorch's `nn.Linear` stores its weight transposed, as `(10, 64)`, computing
  `X @ W.T + b` rather than Step 11's `X @ W + b` — same operation, different storage
  convention.)
- **Time a training run of each framework version against Step 11e's NumPy loop**, on this small
  dataset first, then again after trying [`02_full_mnist.md`](02_full_mnist.md)'s much larger
  one. On this notebook's tiny 1,797-image dataset, plain NumPy is often just as fast or faster —
  the framework's overhead (building a computation graph, autodiff bookkeeping) only pays off once
  the model or data gets large enough, or once a GPU is available to exploit.
- **Try swapping in [`01_hidden_layer_relu.md`](01_hidden_layer_relu.md)'s two-layer network** in
  both frameworks — in Keras, that's one extra `Dense(64, activation="relu")` layer before the
  final one; in PyTorch, `nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))`. Notice
  that you get the chain-rule backpropagation you derived by hand entirely for free this time.

## Related ideas worth knowing about

- **`tf.GradientTape`** / **PyTorch's `autograd`** directly: both frameworks let you drop down to
  a lower level than `model.fit`/`optimizer.step()` and watch gradients get computed step by
  step — a good bridge between Step 11's fully manual `compute_gradients` and the fully automatic
  `model.fit` call above.
- **GPU acceleration**: if you have access to one, rerunning this guide's code with a GPU
  available (both frameworks detect and use one automatically, with no code changes needed) is
  the clearest possible demonstration of the "same core maths, much bigger scale" idea this
  notebook's Step 10 closing note gestures at.
