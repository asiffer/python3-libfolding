from unittest import TestCase

import pyfolding as pf
import numpy as np
from sklearn.datasets import make_blobs


class TestBatchPython(TestCase):
    def test_gaussian_1d(self):
        X = np.random.normal(0, 1, 1000)
        R = pf.FTU(X)
        self.assertTrue(R.n_obs == 1000)
        self.assertTrue(R.dim == 1)
        self.assertTrue(R.folding_statistics > 1)

    def test_gaussian_2d(self):
        X = np.random.multivariate_normal([0, 0],
                                          [[1, 0.5], [0.5, 1]],
                                          2000)
        R = pf.FTU(X)
        self.assertTrue(R.dim == 2)
        self.assertTrue(R.folding_statistics > 1)

    def test_cpp_single_cluster(self):
        # single cluster
        X = make_blobs(3000, 3, 1)[0]
        R = pf.FTU(X, routine="c++")
        print(R)
        print(pf.FTU(X, routine="python"))
        self.assertTrue(R.folding_statistics > 1)

    def test_cpp_two_clusters(self):
        # 2 clusters
        X = make_blobs(5000, 3, 2, random_state=17)[0]
        R = pf.FTU(X, routine="c++")
        print(R)
        self.assertTrue(R.folding_statistics < 1)

    def test_python_cpp_agree_1cluster(self):
        # 1 cluster
        for i in range(10):
            X = make_blobs(500, 2, 1, random_state=i)[0]
            Rcpp = pf.FTU(X, routine="c++")
            Rpy = pf.FTU(X, routine="python")
            err_Phi = abs(Rcpp.folding_statistics -
                          Rpy.folding_statistics) / Rpy.folding_statistics
            self.assertTrue(err_Phi < 0.005)
            err_pivot = np.linalg.norm(Rcpp.folding_pivot - Rpy.folding_pivot)
            self.assertTrue(err_pivot < 1e-6)

    def test_python_cpp_agree_3clusters(self):
        # 3 clusters
        for i in range(10):
            X = make_blobs(900, 2, 3, random_state=i)[0]
            Rcpp = pf.FTU(X, routine="c++")
            Rpy = pf.FTU(X, routine="python")
            err_Phi = abs(Rcpp.folding_statistics -
                          Rpy.folding_statistics) / Rpy.folding_statistics
            self.assertTrue(err_Phi < 0.005)
            err_pivot = np.linalg.norm(Rcpp.folding_pivot - Rpy.folding_pivot)
            self.assertTrue(err_pivot < 1e-6)
