# Probabilistic-Artificial-Intelligence
Project codes for Probabilistic Artificial Intelligence course at ETH Zurich

## Task 1
In this project, Gaussian Process Regressor was used to predict the pollution level at locations where no monitors were placed. To overcome the large data training challenge, the data set was downsampled and the Quadratic Rational Kernel was used to perform training. The second approace was to use the Matern Kernel with nu=1.5 and a fixed lengthscale=15, with which the whole training set can be utilized, resulting in a better outcome. The result was obtained using 99% of the training set. Random Fourier Transform was also tried to tackle the large-scale training challenge but it didn't work well.

## Task 2
This project is about Beyasian Neural Network.

In the first approach, a [Monte Carlo Dropout](https://arxiv.org/pdf/1506.02142.pdf) method was used to perform variational inference of the posterior of the neural network's weights in order to capture the model uncertainty. The neural network was adapted from the MNISTNet, whose input is the flattened 28\*28 images, and the outputs are their estimated labels. The loss was implemented as specified in the paper, namely the cross-entropy loss with a penalty term. The dropout rate was set to 0.1, and the panelty parameter was set to 1e-5. After training, 100 samples were sampled from the trained posterior and performed inference given the input. The inference results were then averaged (Monte Carlo integration) to give and estimated probability of the outputs.

The second approach was to implement the classic variational inference using a Bayesian Neural Network. In the Bayes Net, each layer not only output the pre-activation values for the next layers, but also output the corresponding prior and posterior probability of the weights and the bias. The loss was set as specified in the [paper](https://arxiv.org/pdf/1505.05424.pdf), namely the cross-entropy loss plus the normalized KL-divergence between the posterior and the prior of the weights and bias. Performing loss minimization on this specific loss is the same as doing classic variational inference of the weights and bias, which can capture the model uncertainty. The stored distribution parameters of the weights and bias were used to perform inference, outputing the uncertainty about the estimated digits for the input images.

## Task 3
This task uses Bayesian Optimization to find the maximum accuracy modeled by a Gaussian Process while satisfying the constraint. The speed constraint is also modeled as a GP. The acquisition function is designed to take the speed constraint into account, namely, if the speed of a query point is very likely to violate the constraint, the acquisition function is punished by a large penalty. Doing this makes it very unlikely to query a point that's probable to violate the constraint. In one query, 20 data points are examined and selected by the acquisition function. As new data points are queried, the GP model is updated accordingly, including its mean and covariance. After twenty queries, the one with the highest accuracy value while not violating the constraint is reported as the maximum of the accuracy GP model. 

This algorithm is run for 75 tasks separately, and the average regret is reported as an evaluation.
