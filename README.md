
# Voted Peceptron Experiments

Voted Perceptron experiments under Professor Yoav Freund

Overview: The voted perceptron is an algorithm pioneered by the brilliant 
researchers Yoav Freund and Robert E. Schapire during their time in Bell Labs.
These experiments aim to prove key assumptions regarding the voted perceptron.
More specifically, we hypothesize that while classifying test data using the 
aforementioned algorithm, we can use some **measure of "confidence"** during our
classification, and **only classify** a test instance when the **measured confidence is above
some threshold confidence**. In this repo you will find different experiments 
conducted alongside relevant figures generated from those experiments. 

## Laying the Groundwork

Suppose that we are classifying data into $l$ labels using the voted perceptron.
Then, for each instance, we would calculate a score for each label $s_l$:
```math
s_l = \sum_{i=1}^{k_l}{c_{i}^{l}\text{sign}(\mathbf{v}_{i}^{l} \cdot \mathbf{x})}
```
where:
1. $k_l$ represents $k$ errors for label $l$
2. $c_{i}^{l}$ represents the $c$ correct training samples before the 
$i$-th error (and after the $(i-1)$-th error) for label $l$
3. $\mathbf{v}_{i}^{l}$ represents the prediction vector resulting from the 
voted perceptron algorithm before before the $i$-th error (and after the 
$(i-1)$-th error) for label $l$
4. $\mathbf{x}$ represents the new test instance being classified

## Measures of Confidence

So far, we have 2 measures of confidence:
1. The absolute value of the maximum score (i.e. $\max_{\{l \in \mathbb{Z}, 0 \le l \le 9\}}\left|s_l\right|$)
2. \# of past instances with the same classification as the current
instance $+1$ ($+1$ counts the current instance)

## Repository Navigation

* `NOTEBOOK.md` (File) / `notebooks` (Directory): Describes the purpose, contents, and directions of usage for the `*.ipynb` files inside of the `notebooks` directory
* `TODO.md` (File): A primitive way to view our current tasks (instead of GitHub issues/projects)
* `src` (Directory): Contains the extracted functions we deem to be essential to conduct our experiments. These will also (hopefully) be tested.
* `figures` (Directory): Contains any saved figures from our experiments  
* `tests` (Directory): Contains tests using the pytest framework.

## Tests (pytest)

### Testing Environment Setup

1. Install latest version of `pytest`:
    - `pip install -U pytest`
    - `pytest --version`
2. Run `pip install --editable .[test]` inside the project root directory.
    - Extra documentation below:
    - [Editable installations](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)
    - [pyproject.toml explanation](https://hatch.pypa.io/1.16/config/build/#packages)

### Running Tests

1. Run `pytest -q` to run all tests quietly.
2. Run `pytest -q <test_file.py>` to run a specific test.


### Creating Tests

Please view this `pytest` quickstart [link](https://docs.pytest.org/en/stable/getting-started.html#getstarted)
to get started.
