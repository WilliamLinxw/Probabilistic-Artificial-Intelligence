import numpy
from scipy.stats import laplace, norm, t
import scipy
import math
import numpy as np

VARIANCE = 2.0

normal_scale = math.sqrt(VARIANCE)
student_t_df = (2 * VARIANCE) / (VARIANCE - 1)
laplace_scale = VARIANCE / 2

HYPOTHESIS_SPACE = [norm(loc=0.0, scale=math.sqrt(VARIANCE)),
                    laplace(loc=0.0, scale=laplace_scale),
                    t(df=student_t_df)]

PRIOR_PROBS = np.array([0.35, 0.25, 0.4])


def generate_sample(n_samples, seed=None):
    """ data generating process of the Bayesian model """
    random_state = np.random.RandomState(seed)
    hypothesis_idx = np.random.choice(3, p=PRIOR_PROBS)
    dist = HYPOTHESIS_SPACE[hypothesis_idx]
    return dist.rvs(n_samples, random_state=random_state)


""" Solution """

from scipy.special import logsumexp


def log_posterior_probs(x):
    """
    Computes the log posterior probabilities for the three hypotheses, given the data x

    Args:
        x (np.ndarray): one-dimensional numpy array containing the training data
    Returns:
        log_posterior_probs (np.ndarray): a numpy array of size 3, containing the Bayesian log-posterior probabilities
                                          corresponding to the three hypotheses
    """
    assert x.ndim == 1

    # TODO: enter your code here
    sigma = np.sqrt(2)
    b = 1
    df = 4

    p_H1 = 0.35
    p_X_H1 = norm.pdf(x, 0, sigma)
    p_total_1 = np.append(p_X_H1, p_H1)
    log_p_1 = np.log(p_total_1)
    log_prior_likelihood_1 = np.sum(log_p_1)

    p_H2 = 0.25
    p_X_H2 = laplace.pdf(x, 0, b)
    p_total_2 = np.append(p_X_H2, p_H2)
    log_p_2 = np.log(p_total_2)
    log_prior_likelihood_2 = np.sum(log_p_2)

    p_H3 = 0.4
    p_X_H3 = t.pdf(x, df)
    p_total_3 = np.append(p_X_H3, p_H3)
    log_p_3 = np.log(p_total_3)
    log_prior_likelihood_3 = np.sum(log_p_3)

    log_normalizer = np.log(np.exp(np.sum(log_p_1)) + np.exp(np.sum(log_p_2)) + np.exp(np.sum(log_p_3)))

    log_p1 = log_prior_likelihood_1 - log_normalizer
    log_p2 = log_prior_likelihood_2 - log_normalizer
    log_p3 = log_prior_likelihood_3 - log_normalizer

    log_p = np.array([log_p1, log_p2, log_p3])

    assert log_p.shape == (3,)
    return log_p


def posterior_probs(x):
    return np.exp(log_posterior_probs(x))


""" """


def main():
    """ sample from Laplace dist """
    dist = HYPOTHESIS_SPACE[1]
    x = dist.rvs(1000, random_state=28)

    print("Posterior probs for 1 sample from Laplacian")
    p = posterior_probs(x[:1])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 50 samples from Laplacian")
    p = posterior_probs(x[:50])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 1000 samples from Laplacian")
    p = posterior_probs(x[:1000])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior for 100 samples from the Bayesian data generating process")
    x = generate_sample(n_samples=100)
    p = posterior_probs(x)
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))


if __name__ == "__main__":
    main()
