# NumPy Fundamentals: From Python Lists to Neural Networks

A self-contained Jupyter notebook that teaches **NumPy** — Python's core library for working with
arrays of numbers — from first principles up to an undergraduate ("degree level 5") depth. It's
written as a **learning exercise**, not just a reference: every step includes plain-English
explanations of *what* the code does and *why*, with worked numeric examples throughout.

It starts with plain Python lists and builds up, step by step, to implementing and training a
real (if small) neural network **from scratch, using nothing but NumPy**, on real handwritten
digit images — no other machine learning framework required. Advanced ideas introduced along the
way (derivatives, gradients, the chain rule, softmax, cross-entropy loss) are each explained in
plain English before any formula, so no prior calculus background is assumed.

**Part of a series** on machine learning fundamentals, each solving a related problem at a
different level of the stack:

1. **NumPy_Fundamentals** (this notebook) — build a neural network from scratch using nothing but NumPy
2. **[Scikit_Learn_Guide](https://github.com/tjsuk/Scikit_Learn_Guide)** — the same handwritten-digit
   dataset, solved with scikit-learn's toolkit (including its own `MLPClassifier`)
3. **[HandwrittenTensorflow](https://github.com/tjsuk/HandwrittenTensorflow)** — a full digit-recognition
   project built with TensorFlow/Keras
4. **[HandwrittenPyTorch](https://github.com/tjsuk/HandwrittenPyTorch)** — the same digit-recognition
   project built with PyTorch

They're independent and don't require reading in order, but each links back to the others where
the connection is most relevant.

## What's inside

1. **Import NumPy** and see how it differs from plain Python
2. **The problem NumPy solves** — plain Python lists can't do element-wise maths
3. **Meet the NumPy array** — element-wise operations, `dtype`, `shape`, `size`, `ndim`
4. **Why NumPy is so much faster** — vectorisation, backed by an actual timed benchmark (Python
   loop vs `np.sum`)
5. **Creating arrays** — `zeros`, `ones`, `arange`, `linspace`, `eye`, and NumPy's `random` module
6. **Shapes and dimensions** — scalars, vectors, matrices, tensors, and `reshape`
7. **Indexing and slicing** — including boolean masks and fancy indexing
8. **Broadcasting** — how operations combine arrays of different (but compatible) shapes
9. **Aggregating along an axis** — `sum`/`mean`/`argmax` with `axis=0` vs `axis=1`
10. **Dot products and matrix multiplication** (`@`) — the maths behind every neural network
    layer, with a loop-vs-vectorised speed comparison
11. **Capstone**: building and training a small neural network (softmax regression) entirely from
    scratch with NumPy, on 1,797 real handwritten digit images — covering softmax, cross-entropy
    loss, gradients, and gradient descent
12. **Gradient checking** — numerically verifying the gradient formula used in Step 11 is
    actually correct, using nothing more than the basic definition of a derivative

It finishes with a **Summary** recapping the whole notebook and a glossary of key terms
(vectorisation, broadcasting, dot product, gradient, gradient descent, softmax, cross-entropy,
learning rate, chain rule), plus **Ideas to extend** — adding a hidden layer, trying the
full-resolution MNIST dataset, experimenting with learning rates and mini-batches, and comparing
against a real deep learning framework.

## Requirements

- **Python 3.9+**
- pip

## Setup

1. **Clone or download this repository.**

2. **Create a virtual environment** in the project folder:

   ```bash
   python -m venv .venv
   ```

3. **Activate it.**

   Windows (PowerShell/cmd):
   ```bash
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

4. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Set up `nbstripout`**, which this repository uses to strip notebook cell outputs before
   they're committed (so diffs stay readable and outputs never get checked into git):

   ```bash
   nbstripout --install
   ```

   This registers a git filter scoped to this repository only — it doesn't affect any other
   project on your machine.

6. **Register the environment as a Jupyter kernel** (so the notebook can find your installed
   packages):

   ```bash
   python -m ipykernel install --user --name=numpy-fundamentals-venv --display-name "Python (numpy-fundamentals-venv)"
   ```

## Running the notebook

Launch Jupyter Lab **using the venv's own executable** rather than relying on `activate` alone
(if another Python install is earlier on your `PATH`, a plain `jupyter lab` command can silently
launch the wrong environment):

```bash
.venv\Scripts\jupyter-lab.exe
```

(macOS/Linux: `.venv/bin/jupyter-lab`, after activating the venv.)

Open `numpy_fundamentals.ipynb`, and make sure it's using the **"Python
(numpy-fundamentals-venv)"** kernel (in VS Code: click the kernel picker in the top-right of the
notebook; in Jupyter Lab: use the **Kernel → Change Kernel** menu).

Then run the cells from top to bottom, reading the explanations as you go. The whole notebook
(including training the capstone network) runs in well under a minute on a typical CPU. The
handwritten digit dataset used in Step 11 ships with scikit-learn itself, so nothing needs to be
downloaded — the notebook works fully offline.

## Notes

- The capstone (Step 11) deliberately uses a small, lower-resolution dataset (1,797 images,
  8x8 pixels each) instead of full-size MNIST, so every cell runs almost instantly and stays easy
  to inspect. "Ideas to extend" below suggests scaling this up.
- Everything in this notebook is implemented with NumPy alone — no TensorFlow, PyTorch, or other
  machine learning framework is used or required.

## Ideas to extend

- Add a hidden layer (with a ReLU activation) to turn the Step 11 softmax regression into a true
  small neural network
- Try the full-resolution, 70,000-image MNIST dataset instead (e.g. via
  `sklearn.datasets.fetch_openml('mnist_784')`), and compare accuracy and training time
- Experiment with the learning rate in Step 11e, and watch how the loss curve changes
- Split training into mini-batches instead of using the whole training set on every step, and
  compare training speed and stability
- Rebuild the same softmax regression model using a real framework
  ([TensorFlow/Keras](https://github.com/tjsuk/HandwrittenTensorflow) or
  [PyTorch](https://github.com/tjsuk/HandwrittenPyTorch)), and compare it directly against the
  from-scratch NumPy version in this notebook

Each of these has a detailed, step-by-step walkthrough with full explanations and runnable code
in [`ideas_to_extend/`](ideas_to_extend/README.md).
