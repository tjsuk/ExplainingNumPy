# Extending Step 11: Adding a Hidden Layer with ReLU

## What you're extending

[Step 11](../numpy_fundamentals.ipynb) built **softmax regression**: pixels go straight into a
single weight matrix `W` (shape `64 x 10`) and bias `b`, producing 10 class scores in one step
— `scores = X @ W + b`. That's a neural network with zero hidden layers. Adding one hidden
layer, with a **ReLU** activation in between, turns it into a true (if still small) multi-layer
neural network, and is the natural next step once Step 11's single-layer version is working and
verified by Step 12's gradient check.

## Why this matters

A single linear layer (even followed by softmax) can only draw **straight-line decision
boundaries** between classes in pixel space — no matter how you set `W` and `b`, the model can
never represent a relationship where "class A" requires two disconnected regions of pixel-space.
Stacking a second layer with a **non-linear** activation function in between breaks this
limitation: it lets the network learn intermediate features (curves, strokes, loops) before the
final classification step, which is exactly what makes deep learning "deep." Softmax regression's
clean gradient (`predicted - actual`, Step 11d) doesn't survive adding a layer — you now genuinely
need the **chain rule** mentioned in Step 11d to derive it, so this is also the first place this
notebook's calculus explanation gets used for real, rather than as a preview.

**ReLU** (Rectified Linear Unit) is the standard choice of non-linearity here:
`ReLU(z) = max(0, z)` — it passes positive values through unchanged and clips every negative
value to zero. It's popular because it's cheap to compute and its derivative is trivial
(`1` where `z > 0`, `0` where `z <= 0`), which keeps the chain-rule maths below simple.

## How to do it

### 1. Add a second weight matrix and bias for the hidden layer

Pick a hidden layer size — 64 neurons is a reasonable starting point (same as the input size,
but this is an arbitrary choice you're free to experiment with):

```python
n_hidden = 64

rng = np.random.default_rng(seed=2)
W1 = rng.normal(loc=0.0, scale=0.01, size=(n_features, n_hidden))   # (64, 64): input -> hidden
b1 = np.zeros(n_hidden)
W2 = rng.normal(loc=0.0, scale=0.01, size=(n_hidden, 10))           # (64, 10): hidden -> output
b2 = np.zeros(10)
```

### 2. Extend the forward pass with ReLU in between

```python
def relu(z):
    """Element-wise: pass positives through, clip negatives to zero (Step 3's element-wise ops)."""
    return np.maximum(0, z)

def forward_pass_hidden(X, W1, b1, W2, b2):
    """Two-layer forward pass: Dense -> ReLU -> Dense -> softmax."""
    hidden_scores = X @ W1 + b1        # (n_samples, 64) @ (64, 64) + (64,) -> (n_samples, 64)
    hidden_activated = relu(hidden_scores)
    output_scores = hidden_activated @ W2 + b2   # (n_samples, 64) @ (64, 10) + (10,) -> (n_samples, 10)
    return softmax(output_scores), hidden_activated   # keep hidden_activated - needed for backprop below
```

`cross_entropy_loss` from Step 11c is unchanged — it only ever looked at the final probabilities,
regardless of how many layers produced them.

### 3. Derive the gradients with the chain rule

The gradient of the loss with respect to the **output layer** is exactly the same clean formula
as Step 11d (`probs - y_onehot`), because nothing about the final softmax + cross-entropy step
changed. The chain rule is needed to push that error signal *back through* the new hidden layer —
this is literally what "backpropagation" means:

```python
def compute_gradients_hidden(X, hidden_activated, probs, y_onehot, W2):
    """Gradients for a two-layer network, via the chain rule (backpropagation)."""
    n_samples = X.shape[0]

    # Same as Step 11d: how wrong each output was, per class
    output_error = probs - y_onehot                                # (n_samples, 10)
    grad_W2 = hidden_activated.T @ output_error / n_samples         # (64, n_samples) @ (n_samples, 10)
    grad_b2 = np.mean(output_error, axis=0)

    # Chain rule: push the output error backward through W2, then through ReLU's derivative
    error_at_hidden = output_error @ W2.T                            # (n_samples, 10) @ (10, 64)
    relu_derivative = (hidden_activated > 0).astype(float)           # 1 where ReLU passed through, else 0
    hidden_error = error_at_hidden * relu_derivative                 # element-wise (Step 3)

    grad_W1 = X.T @ hidden_error / n_samples                         # (64, n_samples) @ (n_samples, 64)
    grad_b1 = np.mean(hidden_error, axis=0)

    return grad_W1, grad_b1, grad_W2, grad_b2
```

The pattern to notice: `error_at_hidden = output_error @ W2.T` is the chain rule in matrix form —
"how much did each hidden neuron contribute to the output error, given how strongly it's
connected to each output via `W2`?" Multiplying by `relu_derivative` then zeroes out that blame
for any hidden neuron that was already clipped to zero by ReLU (it couldn't have affected the
output either way).

### 4. Retrain with the new forward/backward pass

```python
learning_rate = 0.5
n_steps = 300
loss_history_hidden = []

for step in range(n_steps):
    probs, hidden_activated = forward_pass_hidden(x_train, W1, b1, W2, b2)
    loss = cross_entropy_loss(probs, y_train_onehot)
    grad_W1, grad_b1, grad_W2, grad_b2 = compute_gradients_hidden(
        x_train, hidden_activated, probs, y_train_onehot, W2
    )

    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2

    loss_history_hidden.append(loss)
    if step % 50 == 0 or step == n_steps - 1:
        print(f"Step {step:3d}:  loss = {loss:.4f}")

test_probs, _ = forward_pass_hidden(x_test, W1, b1, W2, b2)
test_predictions = np.argmax(test_probs, axis=1)
print(f"Test accuracy: {np.mean(test_predictions == y_test):.4f}")
```

## What to observe / think about

- **Re-run Step 12's gradient check against this new `compute_gradients_hidden`.** Nudge a single
  entry of `W1` up and down (not just `W2`) and confirm the numerical and analytic gradients
  still agree — this is the same sanity check the main notebook trusts, now applied to a formula
  you derived yourself rather than one handed to you.
- **Compare final test accuracy against Step 11's single-layer result.** On this small, already
  fairly separable 8x8 digits dataset, the improvement from adding a hidden layer may be modest —
  that's expected, and itself an interesting observation about when extra model capacity actually
  helps.
- **Try a bigger or smaller hidden layer** (`n_hidden = 16` vs `256`) and watch how training speed
  and final accuracy change.
- **Try removing the ReLU** (i.e. just doing `hidden_activated = hidden_scores` with no
  non-linearity) and confirm the two-layer network collapses back to behaving like a single linear
  layer — this is a good way to build real intuition for *why* the non-linearity is the essential
  ingredient, not just an add-on.

## Related NumPy tools worth knowing about

- **`np.maximum(0, z)` vs `np.clip(z, 0, None)`**: both compute ReLU; `np.maximum` is the more
  idiomatic choice and slightly clearer about intent.
- **`scipy.special.softmax`**: a ready-made, numerically-stable softmax if you'd rather not keep
  hand-writing Step 11b's version — useful once you trust your own implementation and want less
  boilerplate in later experiments.
