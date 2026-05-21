
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
where
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
instance $+1$ ($+1$ counts the current instance).
