# Extending Step 11e: Experimenting with the Learning Rate

## What you're extending

[Step 11e](../numpy_fundamentals.ipynb)'s training loop fixes `learning_rate = 0.5` and runs
300 steps, without exploring what happens at other values. Step 11d explains the learning rate
in plain English — "the size of each step" taken against the gradient — but the notebook never
actually shows what happens when that step size is too small or too large. This guide does.

## Why this matters

The learning rate is the single hyperparameter with the most dramatic effect on whether training
works at all, and it's worth *seeing* the two failure modes directly rather than only reading
about them:

- **Too small**, and each step barely nudges the weights — the loss decreases correctly, but so
  slowly that training wastes time (or never gets close to convergence within a fixed number of
  steps).
- **Too large**, and each step overshoots the "downhill" direction the gradient pointed in so
  badly that the loss can bounce around or even increase from one step to the next, instead of
  settling down. In the worst cases, the weights blow up entirely (`nan`/`inf` values).

## How to do it

### 1. Wrap Step 11e's training loop in a function

Refactor the loop so it can be rerun cleanly at different learning rates, always starting from
the same fresh weights (using the same `seed=2` as Step 11e, so every run is a fair comparison):

```python
def train(learning_rate, n_steps=300, X=x_train, y_onehot=y_train_onehot):
    """Step 11e's training loop, parameterised by learning rate."""
    rng = np.random.default_rng(seed=2)
    W = rng.normal(loc=0.0, scale=0.01, size=(n_features, 10))
    b = np.zeros(10)

    loss_history = []
    for step in range(n_steps):
        probs = forward_pass(X, W, b)
        loss = cross_entropy_loss(probs, y_onehot)
        grad_W, grad_b = compute_gradients(X, probs, y_onehot)

        W -= learning_rate * grad_W
        b -= learning_rate * grad_b
        loss_history.append(loss)

    return W, b, loss_history
```

### 2. Run it at several learning rates and plot every curve together

```python
import matplotlib.pyplot as plt

learning_rates = [0.01, 0.05, 0.5, 2.0, 5.0]
results = {}

for lr in learning_rates:
    W_lr, b_lr, history = train(learning_rate=lr)
    test_probs = forward_pass(x_test, W_lr, b_lr)
    accuracy = np.mean(np.argmax(test_probs, axis=1) == y_test)
    results[lr] = (history, accuracy)
    print(f"lr={lr:<6} final loss={history[-1]:.4f}   test accuracy={accuracy:.4f}")

plt.figure(figsize=(7, 5))
for lr, (history, _) in results.items():
    plt.plot(history, label=f"lr={lr}")
plt.xlabel("Training step")
plt.ylabel("Cross-entropy loss")
plt.yscale("log")   # log scale makes it much easier to compare curves that diverge wildly
plt.legend()
plt.title("Effect of learning rate on training")
plt.show()
```

The `yscale("log")` matters here: at `lr=5.0` the loss may swing over several orders of
magnitude between steps, which would otherwise squash every other curve flat on a linear axis.

## What to observe / think about

- **`lr=0.01` should still be visibly decreasing loss after 300 steps**, rather than having
  flattened out — a sign it hasn't yet converged, and would keep improving given more steps.
  Try `n_steps=3000` at this learning rate and see whether it eventually catches up to `lr=0.5`'s
  300-step result.
- **`lr=5.0` is likely to show a visibly jagged or even increasing loss curve** — the step size
  is overshooting the gradient's "downhill" direction. If you push the learning rate high enough
  (try `20.0` or `50.0`), you should be able to make the loss diverge to `nan` entirely — a
  useful failure to see on purpose once, in a safe setting, so you recognise it immediately in a
  real project.
- **Find the smallest number of steps that reaches, say, 90% of `lr=0.5`'s final accuracy, for
  each learning rate.** This reframes "what's the best learning rate" as a genuine speed/stability
  trade-off rather than a single right answer — a run that reaches good accuracy in fewer *steps*
  but risks instability isn't strictly better than a slower, steadier one.
- Combine this with [`04_mini_batch_training.md`](04_mini_batch_training.md): mini-batch
  gradients are noisier than full-batch ones, so a learning rate that was stable in Step 11e's
  full-batch loop may need to be lowered once you switch to mini-batches.

## Related ideas worth knowing about

- **Learning rate schedules**: instead of a single fixed value, gradually shrink the learning
  rate as training progresses (e.g. halve it every 100 steps) — this gets the speed of a large
  initial learning rate with the stability of a small one later on, once the loss is closer to
  its minimum.
- **Adaptive optimisers** (Adam, RMSprop, etc., available in every real deep learning framework):
  these adjust the effective learning rate for *each weight individually*, based on that weight's
  own recent gradient history, and are the reason real training code rarely uses the plain fixed
  step from Step 11e in practice. Implementing plain Adam with only NumPy, following the same
  from-scratch spirit as Step 11, is a natural next challenge after this one.
