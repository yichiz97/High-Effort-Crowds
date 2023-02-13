# High-Effort Crowds: Limited Liability via Tournaments

This is the Python implementation of the agent-based model (ABM) experiments of the paper under the same title which was submitted to NeurIPS 22. The goal is to reproduce several commonly used performance measurements and evaluate them based on two aspects with details discussed in the paper.

## Guide of Usage
First, to run our ABM, the required packages include NumPy, SciPy, Random and Matplotlib is required for plotting. All of the experiments are implemented on JuPyter Notebook version 6.1.4. with Anaconda Navigator 1.10.0 and Python 3.8.5.

The role of the Python files are:
* "**Learn Confusion Matrix from Two Datasets.ipynb**" uses two real-world crowdsourcing datasets (see our paper) to estimate the prior of ground truth and agents' confusion matrix;
* "**Performance Measurements and Sample Generation.ipynb**" implements 2 types of spot-checking mechanisms and 5 types of peer prediction mechanisms as performance measurements and simulate the crowdsourcing model to generate samples of their performance scores;
* "**Evaluating Performance Measurements.ipynb**" uses the score samples to learn the performance score distributions, and evaluates each of the performance measurements, and finally plots the figures in our paper.
* "**Linear_vs_Rank-Order.ipynb**" uses the estimated performance score distributions to compute the optimal RO-payments and the payments under the linear payment functions. Then, it further computes the minimum noise required to ganrantee truthfulness, and the minimal payments after adding noise. Plots on comparisons between RO-payment functions and linear payment functions are generated. 

In "**Data**", we include two datasets that we used. In "**Samples**", we include the generated samples of each performance measurement. Because the intermediate results are saved, one can run the above three files in arbitrary orders. We further note that the second file (sample generation) took hours to run.
