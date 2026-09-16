# Extending Step 11e: Mini-Batch Training

## What you're extending

[Step 11e](../numpy_fundamentals.ipynb) is explicit that "every step processes the entire batch
of 1,400 images at once" — this is called **full-batch (or "batch") gradient descent**: one
weight update per pass over the *entire* training set. This guide splits training into
**mini-batches** instead — smaller chunks of the data, with a weight update after each chunk —
which is what almost all real neural network training actually uses.

## Why this matters

Full-batch gradient descent computes one gradient from all 1,400 training images, then makes
exactly one update. Mini-batch training instead makes *several* updates per pass through the
data — one per mini-batch — so the weights improve more often for the same amount of data seen.
It also matters for a reason that isn't obvious from this notebook's small dataset: on real
datasets with millions of images, the entire training set often doesn't fit in memory at once,
so processing it in smaller chunks isn't just faster, it's the only option. There's a genuine
trade-off, though: each mini-batch's gradient is a noisier, less accurate estimate of the "true"
gradient over all the data (Step 11d), since it's based on fewer examples.

## How to do it

### 1. Shuffle and split the training data into batches each epoch

An **epoch** means one full pass through the training set. Shuffling before each epoch (using a
fresh permutation, exactly like `rng.permutation` in Step 11a) matters — without it, the network
would always see the same batch boundaries and could pick up on the arbitrary order the data
happens to be stored in:

```python
def iterate_minibatches(X, y_onehot, batch_size, rng):
    """Yield shuffled mini-batches of (X, y_onehot), covering the data once."""
    n_samples = X.shape[0]
    shuffled_indices = rng.permutation(n_samples)   # same fancy-indexing idea as Step 11a

    for start in range(0, n_samples, batch_size):
        batch_indices = shuffled_indices[start:start + batch_size]
        yield X[batch_indices], y_onehot[batch_indices]
```

### 2. Rewrite the training loop around epochs and mini-batches

```python
# Reset to the same fresh weights as Step 11e, for a fair comparison
rng_init = np.random.default_rng(seed=2)
W = rng_init.normal(loc=0.0, scale=0.01, size=(n_features, 10))
b = np.zeros(10)

learning_rate = 0.5
batch_size = 100
n_epochs = 20   # 20 epochs x 14 batches/epoch = 280 updates, close to Step 11e's 300 steps
rng_shuffle = np.random.default_rng(seed=7)

loss_history_minibatch = []
for epoch in range(n_epochs):
    for X_batch, y_batch in iterate_minibatches(x_train, y_train_onehot, batch_size, rng_shuffle):
        probs = forward_pass(X_batch, W, b)
        loss = cross_entropy_loss(probs, y_batch)
        grad_W, grad_b = compute_gradients(X_batch, probs, y_batch)

        W -= learning_rate * grad_W
        b -= learning_rate * grad_b
        loss_history_minibatch.append(loss)

    if epoch % 4 == 0 or epoch == n_epochs - 1:
        # Report loss on the FULL training set for a fair, batch-size-independent comparison
        full_loss = cross_entropy_loss(forward_pass(x_train, W, b), y_train_onehot)
        print(f"Epoch {epoch:2d}: full-training-set loss = {full_loss:.4f}")

test_probs = forward_pass(x_test, W, b)
print(f"Test accuracy: {np.mean(np.argmax(test_probs, axis=1) == y_test):.4f}")
```

`compute_gradients` and `forward_pass` themselves don't change at all — they already work on
"however many samples are in `X`," whether that's the full 1,400-image training set or a
100-image mini-batch. Only the loop calling them changes.

## What to observe / think about

- **Plot `loss_history_minibatch` against Step 11e's `loss_history`.** Per *update*, the
  mini-batch loss curve will look visibly noisier (jumping up and down step to step) than the
  smooth, steadily-decreasing full-batch curve — that's the noisier gradient estimate mentioned
  above, and it's expected, not a bug.
- **Compare wall-clock time to reach a given loss, not just step count.** Mini-batches make many
  more (noisier) updates per second than one full-batch update; measure with `time.time()`
  (as Steps 4, 10, and 11e all do) whether this notebook's small dataset actually trains *faster*
  in wall-clock terms with mini-batches, or whether the per-batch Python-loop overhead cancels
  out the benefit at this small scale. Then compare again after trying
  [`02_full_mnist.md`](02_full_mnist.md)'s much larger dataset, where the answer is likely to be
  more clear-cut.
- **Try a few different `batch_size` values** (e.g. 10, 100, 1000, and the full 1,400 — which
  should reproduce Step 11e's original full-batch behaviour almost exactly). Smaller batches mean
  noisier but more frequent updates; larger batches approach full-batch behaviour.
- Combine with [`03_learning_rate_experiments.md`](03_learning_rate_experiments.md): a learning
  rate that worked fine for full-batch updates may need to be lowered for mini-batches, since
  noisier gradients combined with a large step size compound the instability seen there.

## Related ideas worth knowing about

- **`np.array_split`**: a ready-made alternative to the manual slicing above for splitting an
  array into (near-)equal chunks, if you'd rather not hand-write `iterate_minibatches`.
- **Momentum**: an extension to plain gradient descent that keeps a running average of recent
  gradients (rather than reacting to only the current mini-batch), which smooths out exactly the
  noisiness observed above. It's a small step from here towards the adaptive optimisers (Adam,
  RMSprop, etc.) mentioned in [`03_learning_rate_experiments.md`](03_learning_rate_experiments.md).
