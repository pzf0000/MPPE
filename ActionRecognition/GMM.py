# -*- coding: utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np


def fitGMM(data, n=None):
    """
    测试 GMM 的用法
    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :param n: 混合个数，默认不输入由程序判断
    :return: 返回聚类结果
    """
    if n is not None:
        clst = mixture.GaussianMixture(n_components=n)
        clst.fit(data)
        return clst
    X, labels_true = data
    tmp = -1.0
    n = 1
    while True:
        clst = mixture.GaussianMixture(n_components=n)
        clst.fit(X)
        predicted_labels = clst.predict(X)
        ARI = adjusted_rand_score(labels_true, predicted_labels)
        print("n=" + str(n) + "\tARI=" + str(ARI))
        n += 1
        if ARI > tmp:
            tmp = ARI
        else:
            clst = mixture.GaussianMixture(n_components=n-2)
            clst.fit(X)
            break
    return clst


def create_data(centers, num=100, std=0.7):
    """
    生成用于聚类的数据集
    :param centers: 聚类的中心点组成的数组。如果中心点是二维的，则产生的每个样本都是二维的。
    :param num: 样本数
    :param std: 每个簇中样本的标准差
    :return: 用于聚类的数据集。是一个元组，第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    """
    X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    # plt.scatter(X[:, 0], X[:, 1], c=labels_true)
    # plt.show()
    return X, labels_true


def getProba(clst, x):
    """
    得到点在聚类范围内的概率总和
    :param clst: 拟合后的结果
    :param x: 待测值
    :return: 概率总和
    """
    r = np.zeros_like(x[:, 0], dtype=float)
    n = clst.n_components
    for i in range(n):
        mu1 = clst.means_[i][0]
        mu2 = clst.means_[i][1]
        cov = clst.covariances_[i]
        sigmma1 = cov[0][0]
        sigmma2 = cov[1][1]
        a = 1 / (2 * np.pi * sigmma1 * sigmma2)
        b = np.power((x[:, 0] - mu1), 2) / np.power(sigmma1, 2)
        c = np.power((x[:, 1] - mu2), 2) / np.power(sigmma2, 2)
        N = a * np.exp((b + c) / (-2.0))
        r += N
    return r


if __name__ == '__main__':
    centers = [[1, 2], [7, 9], [10, 2]]
    X, labels_true = create_data(centers, 1000, 0.5)
    r = fitGMM((X, labels_true))
    # print(r)
    w = getProba(r, np.array([[1, 2.5], [3.5, 4], [5, 7]]))
    print(w)
