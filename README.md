# Exploring Double Descent
## About
This project is built to explore how ridge regression with optimal regularization behaves on datasets with non-gaussian noise applied to the training data labels. This is mostly to observe if the phenomena described as "double descent" can be mitigated under these alternative parameters.

## Double Descent
The double descent phenomena is described as a critical interval where adding more data causes the test error to rise, then re-descend. 

[Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/pdf/1912.02292.pdf)

## Features
* Calculate an optimal lamba (for regression) as described in [Optimal Regularization Can Mitigate Double Descent](https://arxiv.org/pdf/2003.01897.pdf).
* Plotting the 2D models
* Store results in xlsx file. 
* Batch generation
* Gaussian & Non gaussian noise methods



