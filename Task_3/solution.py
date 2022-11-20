import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
NOISE = 1e-4
SEED = 6
np.random.seed(SEED)

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.accuracy_GP = GaussianProcessRegressor(kernel=0.5*Matern(length_scale=0.5, nu=2.5), random_state=0,
                                                        alpha=1e-5)
        self.speed_GP = GaussianProcessRegressor(kernel=ConstantKernel(constant_value=1.5)+
                                                    np.sqrt(2)*Matern(length_scale=0.5, nu=2.5), random_state=0,
                                                    alpha=1e-5)
        self.points_explored = []



    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        # if not self.previous_points: # At the beginning, choose one sample uniformly at random
        #     next_to_sample = np.array([[np.random.uniform(domain[0][0], domain[0][1])]])
        # else:
        #     next_to_sample = self.optimize_acquisition_function()
        
        return self.optimize_acquisition_function()


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        # Get the input's probability
        s_mean, s_std = self.speed_GP.predict(X=x.reshape(1,-1), return_std=True)
        a_mean, a_std = self.accuracy_GP.predict(X=x.reshape(1,-1), return_std=True)
        # print(c_mean, c_std)
        # print(a_mean, a_std)

        s_mean, s_std = s_mean[0], s_std[0]
        a_mean, a_std = a_mean[0], a_std[0]

        ucb = (a_mean + 3*a_std) - 1e10*(s_mean+2*s_std<SAFETY_THRESHOLD)
        return ucb
        # points = np.array(self.points_explored)
        # if len(points) != 0:
        #     best_so_far = np.max(points[:,1])
        # z = (a_mean - best_so_far)/a_std
        # ei = (a_mean - best_so_far)*norm.cdf(z) + a_std*norm.pdf(z)
        # probability_feasible = 1 - norm.cdf(1.2, loc=s_mean, scale=s_std)
        # weighted_ei = ei*probability_feasible
        # return weighted_ei



    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        if x.shape == (1,):
            x = np.array([[x[0]]])
        self.points_explored.append([float(x[:,0]), float(f), float(v)])
        # print((np.array(self.points_explored)[:,0], np.array(self.points_explored)[:,1]))
        # print("------Accuracy FIT------")
        # print('x shape: ', np.array(self.points_explored)[:,0].reshape(-1,1).shape)
        # print('f shape: ', np.array(self.points_explored)[:,1].shape)
        self.accuracy_GP.fit(np.array(self.points_explored)[:,0].reshape(-1,1), np.array(self.points_explored)[:,1])
        # print("-------SPEED FIT--------")
        # print('x shape: ', np.array(self.points_explored)[:,0].reshape(-1,1).shape)
        # print('v shape: ', np.array(self.points_explored)[:,2].shape)
        self.speed_GP.fit(np.array(self.points_explored)[:,0].reshape(-1,1), np.array(self.points_explored)[:,2])

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        total_points = np.array(self.points_explored)
        feasible_index = np.where(total_points[:,2]>SAFETY_THRESHOLD+3*NOISE)[0]
        # print(feasible_index)
        if len(feasible_index) == 0:
            print('Non-feasible!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            return np.array([0])
        points_feasible = total_points[feasible_index, :]
        max_x = points_feasible[np.argmax(points_feasible[:,1])][0]
        return max_x

""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()