import numpy as np
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder


class GELMClassifier(BaseEstimator, ClassifierMixin):
    '''
    Discriminative graph regularized Extreme Learning Machine (GELM) classifier.
    Implemented based on Yong Peng et al.'s proposed model [1].

    [1] Y. Peng, S. Wang, X. Long, and B. L. Lu, “Discriminative graph regularized 
    extreme learning machine and its application to face recognition,” Neurocomputing, 
    vol. 149, no. Part A, pp. 340-353, Feb. 2015, doi: 10.1016/J.NEUCOM.2013.12.065.

    Parameters
    ----------
    n_hidden: int or 'auto', default='auto'
        If int, the number of hidden nodes. If 'auto', set number of hidden nodes 
        as ten times the size of numbre of features. Any value other than integers
        will be treated as 'auto'.
    
    activation: str, default='leakyrelu'
        Options are 'sigmoid', 'tanh', 'relu' and 'leakyReLU', case-insensitive. 
        When using leakyReLU, "magnitude" parameter must be defined.

    magnitude: float, default=0.1
        The slope magnitude for leakyReLU activation function.
    '''

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def relu(self, x):
        return np.maximum(0.0, x)

    def leakyReLU(self, x, magnitude=0.1):
        return np.maximum(magnitude*x, x)

    def __init__(self, n_hidden='auto', l1=2**10, l2=2**-5, activation='leakyrelu', 
                 magnitude=0.1, random_state=None):
        self.n_hidden = n_hidden
        self.l1 = l1
        self.l2 = l2
        self.activation = activation
        self.magnitude = magnitude
        self.random_state = random_state

        self._activation_funcs = {
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'relu': self.relu,
            'leakyrelu': partial(self.leakyReLU, magnitude=self.magnitude)
        }
        self.activation_func_ = self._activation_funcs[activation.lower()]

    def hidden_nodes(self, X):
        '''
        One forward pass to obtain hidden nodes outputs.

        Parameters
        ----------
        X : ndarray of shape (N, d)
            The input array. Here, N is the number of instances, and d is the number 
            of features.

        Returns
        -------
        H : ndarray of shape (K, N)
            The hidden nodes output, where K is the number of hidden nodes.
        '''
        G = np.dot(self.w_, np.transpose(X))
        G = G + self.b_[:, np.newaxis]
        H = self.activation_func_(G)
        return H

    def laplacian(self, y):
        '''
        Compute graph laplacian from labels y.

        Parameters
        ----------
        y : ndarray of shape (N,)
            The labels for X, where N is the number of instances.

        Returns
        -------
        L : ndarray of shape (N, N)
            The graph laplacian, where N is the number of instances.
        '''
        y_ = y[np.newaxis]
        W = np.zeros([len(y), len(y)])
        for c in np.unique(y):
            y_c = y_ == c
            W += np.dot(np.transpose(y_c), y_c)/np.sum(y_c)
        D = np.diag(np.sum(W, axis=1))
        return D-W

    def fit(self, X, y):
        '''
        Fit the GELM given the training data.

        Parameters
        ----------
        X : ndarray of shape (N, d)
            The input array. Here, N is the number of instances, and d is the number 
            of features.
        y : ndarray of shape (N,)
            The labels for X, where N is the number of instances.

        Returns
        -------
        self
            Fitted estimator.
        '''
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.onehotenc_ = OneHotEncoder()
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = np.unique(y).shape[0]

        self.k_ = self.n_hidden if type(self.n_hidden) == int else 10*self.n_features_in_
        self.w_ = np.random.default_rng(seed=self.random_state)\
                    .normal(size=[self.k_, self.n_features_in_])
        self.b_ = np.random.default_rng(seed=self.random_state).normal(size=[self.k_])
        H = self.hidden_nodes(X)

        # compute graph Laplacian
        L = self.laplacian(y)
        
        # compute beta using GEML's regularization term (equation 14 in [1])
        T = self.onehotenc_.fit_transform(y.reshape(-1, 1)).toarray().transpose()
        HHT = np.dot(H, H.transpose())
        HLHT = np.linalg.multi_dot([H, L, H.transpose()])
        I = np.diag([1]*H.shape[0])
        self.beta_ = np.linalg.multi_dot([np.linalg.inv(HHT + self.l1*HLHT + self.l2*I), 
                                          H, T.transpose()])
        # print(max(self.beta_), self.beta_.min)
        # sorted_beta = np.sort(-abs(self.beta_.flatten()))
        # threshold_idx = int(0.7*len(sorted_beta))
        # threshold_val = abs(sorted_beta[threshold_idx])
        # self.beta_ = np.where(-threshold_val<self.beta_<threshold_val, 0, self.beta_)

        return self

    def predict(self, X):
        '''
        Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray of shape (N, d)
            The input array. Here, N is the number of instances, and d is the number 
            of features.

        Returns
        -------
        y_pred : ndarray of shape (N,)
            Vector containing the class labels for each sample.
        '''
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        TT = np.dot(self.beta_.transpose(), self.hidden_nodes(X)).transpose()
        return np.squeeze(self.onehotenc_.inverse_transform(TT))
