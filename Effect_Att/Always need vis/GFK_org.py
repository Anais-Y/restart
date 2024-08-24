# encoding=utf-8
"""
    Created on 17:25 2018/11/13 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
import bob.learn
import bob.learn.linear
import bob.math
from sklearn.neighbors import KNeighborsClassifier
from GELM import GELMClassifier
from scipy.io import loadmat

class GFK:
    def __init__(self, dim=20):
        '''
        Init func
        :param dim: dimension after GFK
        '''
        self.dim = dim
        self.eps = 1e-20

    def fit(self, Xs, Xt, norm_inputs=None):
        '''
        Obtain the kernel G
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :param norm_inputs: normalize the inputs or not
        :return: GFK kernel G
        '''
        if norm_inputs:
            source, mu_source, std_source = self.znorm(Xs)
            target, mu_target, std_target = self.znorm(Xt)
        else:
            mu_source = np.zeros(shape=(Xs.shape[1]))
            std_source = np.ones(shape=(Xs.shape[1]))
            mu_target = np.zeros(shape=(Xt.shape[1]))
            std_target = np.ones(shape=(Xt.shape[1]))
            source = Xs
            target = Xt

        Ps = self.train_pca(source, mu_source, std_source, 0.99)
        Pt = self.train_pca(target, mu_target, std_target, 0.99)
        Ps = np.hstack((Ps.weights, scipy.linalg.null_space(Ps.weights.T)))
        print("1Ps.shape, Pt.shape:", Ps.shape, Pt.shape)
        Pt = Pt.weights[:, :self.dim]
        print("2Ps.shape, Pt.shape:", Ps.shape, Pt.shape)
        N = Ps.shape[1]
        dim = Pt.shape[1]

        # Principal angles between subspaces
        QPt = np.dot(Ps.T, Pt)
        print(QPt.shape, 'Qpt')
        # [V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
        A = QPt[0:dim, :].copy()
        B = QPt[dim:, :].copy()

        # Equation (2)
        [V1, V2, V, Gam, Sig] = bob.math.gsvd(A, B)
        V2 = -V2

        # Some sanity checks with the GSVD
        I = np.eye(V1.shape[1])
        I_check = np.dot(Gam.T, Gam) + np.dot(Sig.T, Sig)
        assert np.sum(abs(I - I_check)) < 1e-10

        theta = np.arccos(np.diagonal(Gam))

        # Equation (6)
        B1 = np.diag(0.5 * (1 + (np.sin(2 * theta) / (2. * np.maximum
        (theta, 1e-20)))))
        B2 = np.diag(0.5 * ((np.cos(2 * theta) - 1) / (2 * np.maximum(
            theta, self.eps))))
        B3 = B2
        B4 = np.diag(0.5 * (1 - (np.sin(2 * theta) / (2. * np.maximum
        (theta, self.eps)))))

        # Equation (9) of the suplementary matetial
        delta1_1 = np.hstack((V1, np.zeros(shape=(dim, N - dim))))
        delta1_2 = np.hstack((np.zeros(shape=(N - dim, dim)), V2))
        delta1 = np.vstack((delta1_1, delta1_2))

        delta2_1 = np.hstack((B1, B2, np.zeros(shape=(dim, N - 2 * dim))))
        delta2_2 = np.hstack((B3, B4, np.zeros(shape=(dim, N - 2 * dim))))
        delta2_3 = np.zeros(shape=(N - 2 * dim, N))
        delta2 = np.vstack((delta2_1, delta2_2, delta2_3))

        delta3_1 = np.hstack((V1, np.zeros(shape=(dim, N - dim))))
        delta3_2 = np.hstack((np.zeros(shape=(N - dim, dim)), V2))
        delta3 = np.vstack((delta3_1, delta3_2)).T

        delta = np.linalg.multi_dot([delta1, delta2, delta3])
        print("3Ps.shape:", Ps.shape)
        G = np.linalg.multi_dot([Ps, delta, Ps.T])
        print("G.shape:", G.shape)
        sqG = scipy.real(scipy.linalg.fractional_matrix_power(G, 0.5))
        Xs_new, Xt_new = np.dot(sqG, Xs.T).T, np.dot(sqG, Xt.T).T
        return G, Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Fit and use 1NN to classify
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy, predicted labels of target domain, and G
        '''
        G, Xs_new, Xt_new = self.fit(Xs, Xt)
        print('shape after GFK:', Xs_new.shape, Xt_new.shape)
        clf = GELMClassifier(l1=2**0, l2=2**10, random_state=42)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = np.mean(y_pred == Yt.ravel())
        return acc, y_pred, G

    def principal_angles(self, Ps, Pt):
        """
        Compute the principal angles between source (:math:`P_s`) and target (:math:`P_t`) subspaces in a Grassman which is defined as the following:

        :math:`d^{2}(P_s, P_t) = \sum_{i}( \theta_i^{2} )`,

        """
        # S = cos(theta_1, theta_2, ..., theta_n)
        _, S, _ = np.linalg.svd(np.dot(Ps.T, Pt))
        thetas_squared = np.arccos(S) ** 2

        return np.sum(thetas_squared)

    def train_pca(self, data, mu_data, std_data, subspace_dim):
        '''
        Modified PCA function, different from the one in sklearn
        :param data: data matrix
        :param mu_data: mu
        :param std_data: std
        :param subspace_dim: dim
        :return: a wrapped machine object
        '''
        t = bob.learn.linear.PCATrainer()
        machine, variances = t.train(data)

        # For re-shaping, we need to copy...
        variances = variances.copy()

        # compute variance percentage, if desired
        if isinstance(subspace_dim, float):
            cummulated = np.cumsum(variances) / np.sum(variances)
            for index in range(len(cummulated)):
                if cummulated[index] > subspace_dim:
                    subspace_dim = index
                    break
            subspace_dim = index
        machine.resize(machine.shape[0], subspace_dim)
        machine.input_subtract = mu_data
        machine.input_divide = std_data

        return machine

    def znorm(self, data):
        """
        Z-Normaliza
        """
        mu = np.average(data, axis=0)
        std = np.std(data, axis=0)
        data = (data - mu) / std
        return data, mu, std

    def subspace_disagreement_measure(self, Ps, Pt, Pst):
        """
        Get the best value for the number of subspaces
        For more details, read section 3.4 of the paper.
        **Parameters**
          Ps: Source subspace
          Pt: Target subspace
          Pst: Source + Target subspace
        """

        def compute_angles(A, B):
            _, S, _ = np.linalg.svd(np.dot(A.T, B))
            S[np.where(np.isclose(S, 1, atol=self.eps) == True)[0]] = 1
            return np.arccos(S)

        max_d = min(Ps.shape[1], Pt.shape[1], Pst.shape[1])
        alpha_d = compute_angles(Ps, Pst)
        beta_d = compute_angles(Pt, Pst)
        d = 0.5 * (np.sin(alpha_d) + np.sin(beta_d))
        return np.argmax(d)


def read_data(file_path, keyword):
    all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    data = loadmat(file_path)
    tot_data = None
    tot_labels = []
    for i in np.arange(1, 16):
        trial_signal = data[f'{keyword}{i}']
        trial_length = trial_signal.shape[1]
        # print(trial_length)
        if tot_data is None:
            tot_data = trial_signal
        else:
            tot_data = np.concatenate((tot_data, trial_signal), axis=1)
        tot_labels.extend([all_label[i-1]] * trial_length)
    tot_labels = np.array(tot_labels)
    return tot_data, tot_labels


if __name__ == '__main__':
    # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    # for i in range(4):
    #     for j in range(4):
    #         if i != j:
    #             src, tar = 'data/' + domains[i], 'data/' + domains[j]
    #             src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    #             print('source:', src_domain['feas'].shape, src_domain['label'].shape,
    #                   '\ntarget:', tar_domain['feas'].shape, tar_domain['label'].shape,
    #                   '\n label:', np.unique(tar_domain['label']), np.unique(src_domain['label']))
    #             Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
    #             gfk = GFK(dim=20)
    #             acc, ypred, G = gfk.fit_predict(Xs, Ys, Xt, Yt)
    #             print(f'source = {domains[i]}, target = {domains[j]} \n', f'accuracy={acc}')
    data_name1 = '10_20131204.mat'
    file_path1 = f'../../Dataset/SEED/SEED/ExtractedFeatures/{data_name1}'
    data_name2 = '2_20140404.mat'
    file_path2 = f'../../Dataset/SEED/SEED/ExtractedFeatures/{data_name2}'
    tmp_X1, Ys = read_data(file_path1, 'de_LDS')
    tmp_X2, Yt = read_data(file_path2, 'de_LDS')
    channels, time1, bands = tmp_X1.shape
    _, time2, _ = tmp_X2.shape
    print(Ys.shape, Ys[:10])
    X1 = np.transpose(tmp_X1, (1, 0, 2))
    X2 = np.transpose(tmp_X2, (1, 0, 2))
    Xs = X1.reshape((time1, -1))
    Xt = X2.reshape((time2, -1))
    print(Xs.shape, Xt.shape)
    gfk = GFK(dim=20)
    acc, ypred, G = gfk.fit_predict(Xs, Ys, Xt, Yt)
    print(f'source = {data_name1}, target = {data_name2} \n', f'accuracy={acc}')

