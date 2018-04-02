import argparse
import numpy as np
from sklearn.externals import joblib
import ActionRecognition.GMM as GMM


def test_one_joint_in_a_frame(data, clst):
    """
    测试某一帧数据一个关节运动变化是否符合
    :param data: 视频某一帧的运动变化量[坐标]
    :param clst: 高斯拟合后的结果[clst]
    :return: 单关节一帧的概率float
    """
    data = np.array([data])
    r = GMM.getProba(clst, data)
    return r


def test_joints_in_a_frame(data, clst):
    """
    测试某一帧数据18个关节运动变化是否符合
    :param data: 各关节运动变化量[关节标号][坐标]
    :param clst: 高斯拟合后的结果[关节标号][clst]
    :return: 各关节的概率序列[关节标号]
    """
    r = []
    for joint in range(18):
        r.append(test_one_joint_in_a_frame(data[joint], clst[joint]))
    r = np.array(r)
    return r


def test_all_frame(data, clst):
    """
    测试所有帧数据关节运动变化是否符合
    :param data: 各关节运动变化量[帧][关节标号][坐标]
    :param clst: 高斯拟合后的结果[帧][关节标号][clst]
    :return: 所有帧的概率[帧][关节标号]
    """
    len1 = len(data)
    len2 = len(clst)
    frame_num = min(len1, len2)
    r = []
    for frame_index in range(frame_num):
        r.append(test_joints_in_a_frame(data[frame_index][0], clst[frame_index]))
    r = np.array(r)
    return r


def get_joints_proba(proba, threshold=0.9974):
    """
    综合分析所有帧不同关节的概率
    :param proba: 所有帧的概率[帧][关节标号]
    :param threshold: 阈值，默认使用3sigma原则
    :return: 总概率[关节标号]
    """
    t = 1.0 - threshold
    r = []
    for joint_index in range(18):
        tmp = 0.0
        n = 0
        for frame_index in range(len(proba)):
            if proba[frame_index][joint_index] > t:
                tmp += proba[frame_index][joint_index]
                n += 1
        if n != 0:
            tmp = tmp / n
        r.append(float(tmp))
    r = np.array(r)
    return r


def get_all_proba(proba, threshold=0.9974):
    """
    综合分析所有帧所有关节的概率
    :param proba: 所有帧的概率[关节标号]
    :param threshold: 阈值，默认使用3sigma原则
    :return: 总概率
    """
    t = 1.0 - threshold
    n = 0
    tmp = 0.0
    for joint_index in range(18):
        if proba[joint_index] > t:
            tmp += proba[joint_index]
            n += 1
    tmp = tmp / n
    return tmp


if __name__ == '__main__':
    clst_list = joblib.load("boxing.m")
    # [帧][人物][关节][坐标]
    boxing_data = np.load("result/person02_boxing_d2_uncomp.npy")
    walking_data = np.load("result/person05_walking_d1_uncomp.npy")

    p1 = test_all_frame(boxing_data, clst_list)
    p2 = test_all_frame(walking_data, clst_list)

    p11 = get_joints_proba(p1)
    p21 = get_joints_proba(p2)

    p12 = get_all_proba(p11)
    p22 = get_all_proba(p21)

    print(p11)
    print(p21)

    print(p12)
    print(p22)