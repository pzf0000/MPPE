import os
import numpy as np
import matplotlib as plt


def isBosing(pose_delta):
    pass


if __name__ == '__main__':
    folder_path = "camera"
    person_pose_list = []
    person_pose_num = 0
    for (root, dirs, files) in os.walk(folder_path):
        for i in range(len(files)):
            npy_path = os.path.join(root, files[i])
            tmp = np.load(npy_path)
            person_pose_list.append(tmp)
            t = len(tmp)
            if t > person_pose_num:
                person_pose_num = t

    pose_delta = [[[[0 for d in range(2)] for c in range(18)] for b in range(person_pose_num)] for a in
                  range(len(person_pose_list) - 1)]

    for i in range(0, len(person_pose_list) - 1, 1):
        for j in range(len(person_pose_list[i])):
            for k in range(len(person_pose_list[i][j])):
                pose_delta[i][j][k][0] = person_pose_list[i + 1][j][k][0] - person_pose_list[i][j][k][0]
                pose_delta[i][j][k][1] = person_pose_list[i + 1][j][k][1] - person_pose_list[i][j][k][1]
    pose_delta = np.array(pose_delta)

    # k = 30
    # for i in range(len(pose_delta)):
    #     for j in range(len(pose_delta[i])):
    #         print("*******************")
    #         print(str(i) + "\t" + str(j))
    #         x = pose_delta[i][j]
    #         if x[4][0] > k:
    #             print("右手向左" + str(x[4][0]))
    #         elif x[4][0] < -k:
    #             print("右手向右" + str(-x[4][0]))
    #         if x[4][1] > k:
    #             print("右手向下" + str(x[4][1]))
    #         elif x[4][1] < -k:
    #             print("右手向上" + str(-x[4][1]))
    #         print("--------")
    #         if x[7][0] > k:
    #             print("左手向左" + str(x[7][0]))
    #         elif x[7][0] < -k:
    #             print("左手向右" + str(-x[7][0]))
    #         if x[7][1] > k:
    #             print("左手向下" + str(x[7][1]))
    #         elif x[7][1] < -k:
    #             print("左手向上" + str(-x[7][1]))
