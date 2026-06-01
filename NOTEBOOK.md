# voted-perceptron-binary.ipynb

## Overview
This notebook contains an empirical implementation of the Kernel Voted Perceptron algorithm, applied to the MNIST dataset for one-vs-all binary classification. 

Designed as a framework for theoretical machine learning exploration, the notebook focuses on analyzing margin distributions and identifying "hard examples." Rather than simply outputting final predictions, the pipeline evaluates the internal voting mechanics of the perceptron to see how well different confidence heuristics separate correct predictions from mistakes.

## What it Does
* **One-vs-All Classification:** Trains a binary Voted Perceptron model to distinguish a specific target digit from the rest of the dataset.
* **Kernel Exploration:** Evaluates model performance across multiple polynomial degrees.
* **Confidence Heuristics:** Calculates and compares three distinct internal confidence metrics based on the voting history.
* **Calibration Visualization:** Generates detailed 2x3 plots comparing Error Rate against both Confidence Thresholds and Dataset Coverage.

## How it Works

### 1. Optimized Training Pipeline
The `train_voted_perceptron_binary` function implements the core algorithm. To optimize execution speed during training, it minimizes unnecessary conversions of python List to NumPy arrays. Instead, it pre-allocates memory arrays (`M_x_arr`, `M_y_arr`) to efficiently track mistakes and updates the survival times (the `C` array) for each constituent perceptron.

### 2. Batch Evaluation & Confidence Metrics
The `evaluate_binary` function processes test data in batches to manage memory while calculating weighted kernel matrices. It extracts three specific confidence metrics to analyze the margin of victory:
* **Unbroken Streak:** Measures how long the final models in the sequence agreed by counting the minimum consecutive matching votes.
* **Absolute Vote Difference:** Calculates the raw margin of victory by taking the absolute difference between raw positive and negative votes.
* **Survival-Weighted Confidence:** Calculates the margin of victory scaled by the survival times (weights) of the models casting the votes.

### 3. Visualizing Calibration
The `plot_calibration` function uses the `get_error_rates` helper to enforce sample minimums (preventing random variance artifacts) and plots the performance of the three heuristics. The resulting visualizations show:
* **Error Rate vs. Threshold:** Demonstrating the ideal scenario where the error drops as the threshold increases.
* **Error Rate vs. Coverage:** Demonstrating how the error remains low until the model is forced to predict on the hardest examples (approaching 100% coverage).

## Dependencies
* `numpy`
* `matplotlib`
* `scikit-learn` (for fetching MNIST and data splitting)
* `time`

## Usage
Run the notebook cells sequentially. The main execution block will automatically fetch the MNIST dataset, iterate through digits (0-9), binarize the labels for the specific task, train the Voted Perceptron across defined polynomial degrees, and output the calibration plots for analysis.

# voted-perceptron-for-experiment

## Overview
This notebook contains an extensive empirical evaluation of the Voted Perceptron algorithm and its kernelized variants, applied to the MNIST dataset. 

This notebook acts as a massive benchmarking experiment. It tests a one-vs-all binary classification task for all 10 digits, comparing multiple internal voting heuristics, polynomial degrees, and epoch checkpoints to see which prediction strategy yields the lowest test error.

## What it Does
* **Full-Scale Benchmarking:** Iterates through every digit (0-9) to train and evaluate one-vs-all classifiers.
* **Polynomial Kernel Scaling:** Evaluates model performance across polynomial degrees $d \in [1, 2, 3, 4, 5, 6]$.
* **Epoch Checkpointing:** Evaluates test error at specific training milestones ($T \in [0.1, 1, 2, 3, 4, 10, 30]$ epochs).
* **Heuristic Comparison:** Calculates predictions using 7 different decision rules:
  * Standard Vote
  * Average (Normalized & Unnormalized)
  * Last Perceptron (Normalized & Unnormalized)
  * Random Perceptron (Normalized & Unnormalized)
* **Complexity Tracking:** Logs the number of mistakes made during training and the resulting number of Support Vectors for every configuration.

## How it Works

### 1. Optimized Memory Management
To handle the massive scale of this experiment without severe bottlenecks, the notebook utilizes custom training functions (`train_predict_linear` and `train_predict_kernel`). Instead of using slow Python lists for tracking mistakes, it preallocates large NumPy arrays (`M_x_arr`, `M_y_arr` starting with a capacity of 100,000). This minimizes unnecessary conversions of python List to NumPy arrays.

### 2. Batch Matrix Operations for Prediction
During evaluation, the test set is processed in batches (e.g., `batch_size = 1000`). The notebook computes the weighted kernel matrix for the batch and uses cumulative sums (`np.cumsum`) to efficiently calculate the hypotheses of all intermediate models in the sequence. This allows it to extract the Average, Last, and Vote scores simultaneously.

### 3. Automated Tabulation
The `run_experiment_for_digit` function aggregates the accuracy metrics and outputs them into highly structured Pandas DataFrames. For each digit, it generates two tables:
* **Table 1:** Results for lower polynomial degrees ($d=1, 2, 3$).
* **Table 2:** Results for higher polynomial degrees ($d=4, 5, 6$).
These tables present a clear, grid-like view of Test Error % against the $T$ checkpoints for all heuristics.

## Dependencies
* `numpy`
* `pandas` (for result tabulation)
* `scikit-learn` (for fetching MNIST and train/test splitting)
* `time` (for benchmarking execution speed)

## Usage
Simply run all cells in the notebook. The pipeline will automatically:
1. Fetch and scale the MNIST dataset (60k train / 10k test).
2. Iterate through digits 0 to 9.
3. Train the models, generate the predictions, and display the resulting Pandas DataFrames directly in the cell outputs.

# voted-perceptron-binary.ipynb

## Overview
This notebook contains an empirical implementation of the Kernel Voted Perceptron algorithm, applied to the MNIST dataset for one-vs-all binary classification. 

Designed as a framework for theoretical machine learning exploration, the notebook focuses on analyzing margin distributions and identifying "hard examples." Rather than simply outputting final predictions, the pipeline evaluates the internal voting mechanics of the perceptron to see how well different confidence heuristics separate correct predictions from mistakes.

## What it Does
* **One-vs-All Classification:** Trains a binary Voted Perceptron model to distinguish a specific target digit from the rest of the dataset.
* **Kernel Exploration:** Evaluates model performance across multiple polynomial degrees.
* **Confidence Heuristics:** Calculates and compares three distinct internal confidence metrics based on the voting history.
* **Calibration Visualization:** Generates detailed 2x3 plots comparing Error Rate against both Confidence Thresholds and Dataset Coverage.

## How it Works

### 1. Optimized Training Pipeline
The `train_voted_perceptron_binary` function implements the core algorithm. To optimize execution speed during training, it minimizes unnecessary conversions of python List to NumPy arrays. Instead, it pre-allocates memory arrays (`M_x_arr`, `M_y_arr`) to efficiently track mistakes and updates the survival times (the `C` array) for each constituent perceptron.

### 2. Batch Evaluation & Confidence Metrics
The `evaluate_binary` function processes test data in batches to manage memory while calculating weighted kernel matrices. It extracts three specific confidence metrics to analyze the margin of victory:
* **Unbroken Streak:** Measures how long the final models in the sequence agreed by counting the minimum consecutive matching votes.
* **Absolute Vote Difference:** Calculates the raw margin of victory by taking the absolute difference between raw positive and negative votes.
* **Survival-Weighted Confidence:** Calculates the margin of victory scaled by the survival times (weights) of the models casting the votes.

### 3. Visualizing Calibration
The `plot_calibration` function uses the `get_error_rates` helper to enforce sample minimums (preventing random variance artifacts) and plots the performance of the three heuristics. The resulting visualizations show:
* **Error Rate vs. Threshold:** Demonstrating the ideal scenario where the error drops as the threshold increases.
* **Error Rate vs. Coverage:** Demonstrating how the error remains low until the model is forced to predict on the hardest examples (approaching 100% coverage).

## Dependencies
* `numpy`
* `matplotlib`
* `scikit-learn` (for fetching MNIST and data splitting)
* `time`

## Usage
Run the notebook cells sequentially. The main execution block will automatically fetch the MNIST dataset, iterate through digits (0-9), binarize the labels for the specific task, train the Voted Perceptron across defined polynomial degrees, and output the calibration plots for analysis.

# voted-perceptron-for-experiment

## Overview
This notebook contains an extensive empirical evaluation of the Voted Perceptron algorithm and its kernelized variants, applied to the MNIST dataset. 

This notebook acts as a massive benchmarking experiment. It tests a one-vs-all binary classification task for all 10 digits, comparing multiple internal voting heuristics, polynomial degrees, and epoch checkpoints to see which prediction strategy yields the lowest test error.

## What it Does
* **Full-Scale Benchmarking:** Iterates through every digit (0-9) to train and evaluate one-vs-all classifiers.
* **Polynomial Kernel Scaling:** Evaluates model performance across polynomial degrees $d \in [1, 2, 3, 4, 5, 6]$.
* **Epoch Checkpointing:** Evaluates test error at specific training milestones ($T \in [0.1, 1, 2, 3, 4, 10, 30]$ epochs).
* **Heuristic Comparison:** Calculates predictions using 7 different decision rules:
  * Standard Vote
  * Average (Normalized & Unnormalized)
  * Last Perceptron (Normalized & Unnormalized)
  * Random Perceptron (Normalized & Unnormalized)
* **Complexity Tracking:** Logs the number of mistakes made during training and the resulting number of Support Vectors for every configuration.

## How it Works

### 1. Optimized Memory Management
To handle the massive scale of this experiment without severe bottlenecks, the notebook utilizes custom training functions (`train_predict_linear` and `train_predict_kernel`). Instead of using slow Python lists for tracking mistakes, it preallocates large NumPy arrays (`M_x_arr`, `M_y_arr` starting with a capacity of 100,000). This minimizes unnecessary conversions of python List to NumPy arrays.

### 2. Batch Matrix Operations for Prediction
During evaluation, the test set is processed in batches (e.g., `batch_size = 1000`). The notebook computes the weighted kernel matrix for the batch and uses cumulative sums (`np.cumsum`) to efficiently calculate the hypotheses of all intermediate models in the sequence. This allows it to extract the Average, Last, and Vote scores simultaneously.

### 3. Automated Tabulation
The `run_experiment_for_digit` function aggregates the accuracy metrics and outputs them into highly structured Pandas DataFrames. For each digit, it generates two tables:
* **Table 1:** Results for lower polynomial degrees ($d=1, 2, 3$).
* **Table 2:** Results for higher polynomial degrees ($d=4, 5, 6$).
These tables present a clear, grid-like view of Test Error % against the $T$ checkpoints for all heuristics.

## Dependencies
* `numpy`
* `pandas` (for result tabulation)
* `scikit-learn` (for fetching MNIST and train/test splitting)
* `time` (for benchmarking execution speed)

## Usage
Simply run all cells in the notebook. The pipeline will automatically:
1. Fetch and scale the MNIST dataset (60k train / 10k test).
2. Iterate through digits 0 to 9.
3. Train the models, generate the predictions, and display the resulting Pandas DataFrames directly in the cell outputs.

# multiclass-voted-perceptron.ipynb

## Overview
This notebook contains an empirical implementation of the True Multiclass Kernel Voted Perceptron algorithm (Dual Form), applied to the MNIST dataset for 10-way digit classification. 

Designed as a framework for theoretical machine learning exploration, the notebook focuses on analyzing margin distributions and identifying "hard examples." Rather than simply outputting final predictions, the pipeline evaluates the internal voting mechanics of the perceptron to see how well different confidence heuristics separate correct predictions from mistakes.

## What it Does
* **True Multiclass Classification:** Trains a Multiclass Voted Perceptron model, maintaining a single global timeline of mistakes across all 10 classes, rather than relying on independent one-vs-all classifiers.
* **Kernel Exploration:** Evaluates model performance across multiple polynomial degrees ($d \in [1, 2, 3, 4, 5, 6]$).
* **Confidence Heuristics:** Calculates and compares two distinct internal confidence metrics based on historical voting shifts: Unbroken Streak and Survival-Weighted Margin.
* **Calibration & Distribution Visualization:** Generates detailed calibration profiles comparing Error Rate against Confidence Thresholds, alongside Histograms and Cumulative Distribution Functions (CDFs) of the test set margin distributions.

## How it Works

### 1. Global Timeline & Optimized Memory Management
The `train_voted_perceptron_multiclass` function implements the core algorithm. To optimize execution speed and handle the multiclass update, it avoids slow Python list-append operations. Instead, it pre-allocates static memory arrays (`M_x`, `M_y_true`, `M_y_pred`) to efficiently record a single, global sequence of mistakes and tracks the survival times (the `c_array`) for each historical model.

### 2. Batch Evaluation & Global Argmax Tracking
The `evaluate_multiclass` function processes test data in batches to manage memory while computing weighted kernel matrices. It reconstructs the cumulative scores for all historical models, tracks the global predicted class from every historical step, and extracts two specific confidence metrics:
* **Survival-Weighted Unbroken Streak:** Measures how long the final sequence of models agreed by summing the survival weights of the final unbroken prediction streak.
* **Survival-Weighted Margin:** Calculates the margin of victory by taking the difference between the first and second place survival-weighted vote totals.

### 3. Visualizing Calibration and Distributions
The pipeline uses custom plotting functions (`plot_calibration_multiclass` and `plot_distributions`) to analyze model confidence. The visualizations include:
* **Error Rate vs. Threshold:** Demonstrating how the error drops as the survival-weighted streak or vote margin threshold increases (utilizing a helper function to enforce sample minimums and prevent random variance artifacts).
* **Histograms & CDFs:** Revealing the exact distribution and frequency of unbroken streak lengths and margins of victory across the entire test set.

## Dependencies
* `numpy`
* `matplotlib`
* `scikit-learn` (for fetching MNIST and data splitting)
* `time`

## Usage
Run the notebook cells sequentially. The main execution block will automatically fetch the MNIST dataset, train the True Multiclass Voted Perceptron across defined polynomial degrees, evaluate the test set to determine accuracy, and output the calibration and distribution plots for analysis.
