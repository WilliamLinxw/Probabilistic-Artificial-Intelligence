# Probabilistic-Artificial-Intelligence
Project codes for Probabilistic Artificial Intelligence course at ETH Zurich

## Task 1
In this project, Gaussian Process Regressor was used to predict the pollution level at locations where no monitors were placed. To overcome the large data training challenge, the data set was downsampled and the Quadratic Rational Kernel was used to perform training. The second approace was to use the Matern Kernel with nu=1.5 and a fixed lengthscale=15, with which the whole training set can be utilized, resulting in a better outcome. The result was obtained using 99% of the training set. Random Fourier Transform was also tried to tackle the large-scale training challenge but it didn't work well.

## Task 3
This task uses Bayesian Optimization to find the maximum accuracy modeled by a Gaussian Process while satisfying the constraint. The speed constraint is also modeled as a GP. The acquisition function is designed to take the speed constraint into account, namely, if the speed of a query point is very likely to violate the constraint, the acquisition function is punished by a large penalty. Doing this makes it very unlikely to query a point that's probable to violate the constraint. In one query, 20 data points are examined and selected by the acquisition function. As new data points are queried, the GP model is updated accordingly, including its mean and covariance. After twenty queries, the one with the highest accuracy value while not violating the constraint is reported as the maximum of the accuracy GP model. 

This algorithm is run for 75 tasks separately, and the average regret is reported as an evaluation.
