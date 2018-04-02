import argparse
import numpy as np
from sklearn.externals import joblib

import ActionRecognition.GMM as GMM


def train(npy_path, ):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', '-p')
    # args = parser.parse_args()
    npy = npy_path
    video_num = len(npy)
    frame_min = 10000

    # 运动变化量
    for video in range(video_num):
        for frame in range(len(npy[video]) - 1):
            npy[video][frame][0] = npy[video][frame + 1][0] - npy[video][frame][0]

    # 不同视频帧数不同，取最小值
    for video_index in range(video_num):
        frame_num = len(npy[video_index])
        if frame_num < frame_min:
            frame_min = frame_num

    # 去掉最后一帧
    frame_min -= 1

    # [帧][关节标号][坐标(x, y)]
    gmms = [[0 for _ in range(18)] for _ in range(frame_min)]

    # 取不同视频的相同帧，第0个人物的关节
    for frame_index in range(frame_min):
        for joint_index in range(18):
            points = []
            for video_index in range(video_num):
                point = npy[video_index][frame_index][0][joint_index]
                points.append(point)
            points = np.array(points)
            gmms[frame_index][joint_index] = points
    gmms = np.array(gmms)

    # [帧][关节][clst]
    clst_list = [[0 for _ in range(18)] for _ in range(frame_min)]
    for frame_index in range(frame_min):
        for joint_index in range(18):
            clst = GMM.fitGMM(gmms[frame_index][joint_index], 3)
            clst_list[frame_index][joint_index] = clst
    joblib.dump(clst_list, "ActionRecognition/Actions/boxing.m")


if __name__ == '__main__':
    # [视频文件][帧][人物][关节][坐标]
    # npy = np.load("data/KTH/" + str(args.path) + ".npy")
    npy_path = np.load("data/KTH/boxing.npy")
    train(npy_path)