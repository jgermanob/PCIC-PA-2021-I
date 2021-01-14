import numpy as np
import concurrent.futures
from threading import Thread
from model import Model
from scipy import optimize


# class hklearn:
class LogisticRegression(Model):
    def __init__(self, C = 1.0, n_jobs = None, solver = 'fmincg', maxiter = 50):
        self.C = C
        self.n_jobs = n_jobs
        self.solver = solver
        self.all_theta = []
        self.max_iter = maxiter

    def func(self, thetas_p, max_iter, n, c, X_p, y_p, C):
        initial_theta = np.zeros((n + 1, 1), dtype=np.float64)
        args = [X_p[c], y_p[c], C]
        print('Iter: ', c)
        theta= optimize.fmin_cg(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=max_iter)
        thetas_p[c] = theta.transpose()

    def func2(self, y_p, c, y, X_p, X):
        X_p[c] = X
        y_p[c] = np.array(list(map(lambda x : 1.0 if x == c else 0.0, y)), dtype=np.float64)  


    def fit(self, X, y):
        n_labels = len(set(y))
        n = X.shape[1]
        m = m = X.shape[0]
        self.all_theta = np.zeros((n_labels, n + 1), dtype=np.float64)
        X_aux = np.concatenate((np.ones((m,1), dtype = np.float64), X), axis=1)
        initial_theta = np.zeros((n + 1, 1), dtype=np.float64)
        theta = np.zeros((n + 1, 1), dtype=np.float64)
        args = [X_aux, y.copy(), self.C]


        
        if self.n_jobs is None:
            for c in range(n_labels):
                args[1] = np.array(list(map(lambda x : 1.0 if x == c else 0.0, y)), dtype=np.float64)
                theta= optimize.fmin_cg(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=self.max_iter)
                self.all_theta[c, :] = theta.transpose()
        else:
            y_p = {}
            thetas = {}
            X_p = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers = self.n_jobs) as executor:
                for c in range(n_labels):
                    future = executor.submit(self.func2, y_p, c, y, X_p, X_aux)
                    future = executor.submit(self.func, thetas, self.max_iter, n, c, X_p, y_p, self.C)

            for c in range(n_labels):
                self.all_theta[c,:] = thetas[c]


    def predict(self, X):
        m = X.shape[0]
        num_labels = self.all_theta.shape[0]
        p = np.zeros((m, 1), dtype=np.float64)
        aux = np.zeros((num_labels, 1), dtype=np.float64)
        X_aux = np.concatenate((np.ones((m,1), dtype = np.float64), X), axis=1)
        s = self.sigmoid(np.matmul(X_aux, self.all_theta.transpose()))
        return np.hstack(list(map(lambda x : np.where(x == np.amax(x))[0], s)))

    

    def sigmoid(self, z):
        g = 1./(1. + np.exp(-z, dtype=np.float64))
        return g

    def cost_func(self, theta, *args):
        X, y, C = args
        m = X.shape[0]
        theta_aux = theta.copy()
        theta_aux[0] = 0 
        cost = (1/m)*(np.matmul(-y.transpose(),np.log(self.sigmoid(np.matmul(X,theta, dtype=np.float64))), dtype=np.float64) 
        - np.matmul((1-y).transpose(),np.log(1-self.sigmoid(np.matmul(X,theta, dtype=np.float64))), dtype=np.float64))
        reg_term = (1/(2*C*m))*(np.matmul(theta_aux.transpose(),theta_aux, dtype=np.float64))
        J = cost + reg_term
        return J

    def grad_cost_func(self, theta, *args):
        X, y, C = args
        m = X.shape[0]
        I = np.eye(len(theta))
        I[0,0] = 0
        grad = (1/m)*np.matmul(X.transpose(), (self.sigmoid(np.matmul(X,theta)) - y), dtype=np.float64)+ (1/(C*m))*(np.matmul(I,theta, dtype=np.float64))
        return grad[:]
