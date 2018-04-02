import json
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    path = "boxing"
    # [视频文件][帧][人物][关节][坐标]
    npy = np.load("data/KTH/" + path + ".npy")
    for video in range(len(npy)):
        for frame in range(len(npy[video]) - 1):
            npy[video][frame][0] = npy[video][frame + 1][0] - npy[video][frame][0]

    # fig = plt.figure()
    # ax = Axes3D(fig)
    r = []
    k = 5
    j = 7
    # while k > 0:
    #     r.append(random.randint(0, 99))
    #     k -= 1
    # print(r)
    for i in range(len(npy)):
        x = np.arange(len(npy[i]) - 1) + 1
        y = []
        z = []
        for n in range(len(x)):
            y.append(npy[i][n][0][j][0])
            z.append(npy[i][n][0][j][1])
        # y.append(npy[i][n][0][j][0])
        # z.append(npy[i][n][0][j][1])
        y = np.array(y)
        z = np.array(z)
        # ax.scatter3D(x, y, z)
        # ax.plot3D(y, x * 6, z)
        plt.close()
        plt.plot(y, z, 'r*')
        plt.legend()
        # ax.set_xlabel("x")
        # ax.set_ylabel("Frame")
        # ax.set_zlabel("y")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(path + str(i))
        # plt.show()
        plt.savefig("data/dpt/" + path + str(i))
    print("Done.")