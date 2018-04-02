import numpy as np


def u(D, k, x, mu, sigma):
    a = x - mu[k]
    b = 0
    for x in a:
        b += x ** 2
    c = 2 * sigma[k]
    d = np.power(2 * np.pi, float(D) / 2)
    e = np.sqrt(np.abs(sigma[k]))
    return np.exp((b / c) / (d * e))


def computeGamma(D, t, k, K, omega, mu, sigma):
    sum = 0
    for i in range(K):
        sum += omega[i] * u(D, i, X[t], mu, sigma)
    return (omega[k] * u(D, k, X[t], mu, sigma)) / sum


def computeStatistics(K, T, D, S, X, omega, mu, sigma):
    for k in range(K):
        S[k][0] = 0
        S[k][1] = np.zeros(D)
        S[k][2] = np.zeros(D)

    for t in range(T):
        for k in range(K):
            gamma = computeGamma(D, t, k, K, omega, mu, sigma)
            S[k][0] = S[k][0] + gamma
            S[k][1] = S[k][1] + gamma * X[t]
            S[k][2] = S[k][2] + gamma * X[t] * X[t]
    return S


def computeFisherVectorSignature(K, T, D, S, X, omega, mu, sigma):
    F_alpha = [S[0][0] for _ in range(K)]
    F_mu = [S[0][1] for _ in range(K)]
    F_sigma = [S[0][2] for _ in range(K)]
    for k in range(K):
        F_alpha[k] = (S[k][0] - T * omega[k]) / np.sqrt(omega[k])
        F_mu[k] = (S[k][1] - mu[k] * S[k][0]) / (np.sqrt(omega[k]) * sigma[k])
        F_sigma[k] = (S[k][2] - 2 * mu[k] * S[k][1] + (np.power(mu[k], 2) - np.power(sigma[k], 2)) * S[k][0]) / (
                    np.sqrt(2 * omega[k]) * np.power(sigma[k], 2))
    return F_alpha, F_mu, F_sigma


def applyNormalizations(F):
    """
    归一化
    :param F: Fisher向量
    :return: 归一化后的Fisher向量
    """
    F_alpha = F[0]
    F_mu = F[1]
    F_sigma = F[2]
    sum = 0
    for i in range(len(F_alpha)):
        F_alpha[i] = np.sqrt(np.abs(F_alpha[i])) * np.sign(F_alpha[i])
        sum += F_alpha[i] ** 2
    F_alpha = F_alpha / np.sqrt(sum)

    for i in range(len(F_mu)):
        F_mu[i] = np.sqrt(np.abs(F_mu[i])) * np.sign(F_mu[i])
        sqr = F_mu[i] ** 2
        sum = 0
        for s in sqr:
            sum += s
        F_mu[i] = F_mu[i] / np.sqrt(sum)

    for i in range(len(F_sigma)):
        F_sigma[i] = np.sqrt(np.abs(F_sigma[i])) * np.sign(F_sigma[i])
        sqr = F_sigma[i] ** 2
        sum = 0
        for s in sqr:
            sum += s
        F_sigma[i] = F_sigma[i] / np.sqrt(sum)
    return F_alpha, F_mu, F_sigma


def transposeFv(F):
    """
    转置并重新排列
    :param F: Fisher向量
    :return: F_alpha, F_mu, F_sigma，均为一维向量
    """
    F_alpha = F[0]
    F_mu = F[1]
    F_sigma = F[2]

    tmp = []
    for f in F_alpha:
        tmp.append(f)
    F_alpha = np.array(tmp)

    tmp = []
    for k in F_mu:
        for d in k:
            tmp.append(d)
    F_mu = np.array(tmp)

    tmp = []
    for k in F_sigma:
        for d in k:
            tmp.append(d)
    F_sigma = np.array(tmp)
    return F_alpha, F_mu, F_sigma


def computeFisherVector(X, omega, mu, sigma):
    """
    计算Fisher向量
    :param X: 数据点，维数为D
    :param omega: 每个部分的权重，总和为1
    :param mu: 每个部分的平均值
    :param sigma: 每个部分的方差
    :return: alpha, mu, sigma对应的Fisher向量
    """
    K1 = len(omega)
    K2 = len(mu)
    K3 = len(sigma)
    if K1 != K2 or K1 != K3 or K2 != K3:
        print("Error. Parameter lengths of lambda are different!")
        return None
    K = K1
    T = len(X)
    D = len(X[0])
    # S = [[0 for _ in range(3)] for _ in range(K)]
    S = [[0, [0 for _ in range(D)], [0 for _ in range(D)]] for _ in range(K)]
    S = computeStatistics(K, T, D, S, X, omega, mu, sigma)
    F = computeFisherVectorSignature(K, T, D, S, X, omega, mu, sigma)
    Fv = applyNormalizations(F)
    Fv = transposeFv(Fv)
    return Fv


def mergeF(F):
    """
    将Fisher向量的三个部分合成一个一维向量
    :param F: Fisher向量
    :return: 整合为一维的Fisher向量
    """
    F_alpha = F[0]
    F_mu = F[1]
    F_sigma = F[2]

    tmp = []
    for f in F_alpha:
        tmp.append(f)
    for k in F_mu:
        tmp.append(k)
    for k in F_sigma:
        tmp.append(k)
    F = np.array(tmp)
    return F


if __name__ == '__main__':
    X = np.array([(0, 1, 1), (1, 2, 5), (3, 7, 3)])
    omega = np.array([0.5, 0.5])
    mu = np.array([1, 2])
    sigma = np.array([1, 2])
    v = computeFisherVector(X, omega, mu, sigma)
    vt = mergeF(v)
