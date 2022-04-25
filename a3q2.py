import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import collections


def approximation_algorithm(m, mu, sigma, K):
    # Function replicating algorithm from document, returns list of p_i's
    theta_min = mu - 3 * sigma
    theta_max = mu + 3 * sigma
    theta_range = theta_max - theta_min
    theta_bin_size = theta_range / m
    p = np.zeros(m)

    samples = np.random.normal(mu, sigma, K)
    for sample in samples:
        theta_before = math.floor((sample - theta_min)/theta_bin_size) 
        theta_after = theta_before + 1

        if theta_before < 0:
            p[0] += 1/K

        elif theta_after >= m:
            p[m-1] += 1/K

        else:
            theta_before_pos = theta_before * theta_bin_size + theta_min
            theta_after_pos = theta_after * theta_bin_size + theta_min
            p[theta_before] += 1/K * (-(sample - theta_before_pos))/-theta_bin_size
            p[theta_after] += 1/K * (-(theta_after_pos - sample))/-theta_bin_size
    return p

# Code to plot a histogram of samples from our PMF (normalized to be a valid PMF) vs. the true PDF of the normal distribution
mu = 3
sigma = 2

theta_min = mu - 3 * sigma
theta_max = mu + 3 * sigma
p = approximation_algorithm(100, mu, sigma, 10000)
n_compare_samples = 10000
x = np.linspace(theta_min, theta_max, 100)
approx_samples = np.random.choice(x, n_compare_samples, p=p)
plt.hist(approx_samples, density=True, bins=100, label='Normalized histogram of approximate pmf')

plt.plot(x, stats.norm.pdf(x, mu, sigma), label="True pdf")
plt.title("Normalized histogram of samples from approximate pmf vs. true pdf")
plt.legend()
plt.show()
