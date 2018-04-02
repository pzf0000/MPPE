import time

import chainer
import cv2
import numpy as np

from PosturalRecognition.train.pose_detector import PoseDetector

chainer.using_config('enable_backprop', False)


def calpos(img, q):
    pose_detector = PoseDetector("posenet", "../models/posenet.npz")
    person_pose_array = pose_detector(img, fast_mode=True)
    x = np.array(person_pose_array)
    np.save("result/" + str(time.time()) + ".npy", x)


def main(frequency=5, weight=640, height=480):
    pose_detector = PoseDetector("posenet", "PosturalRecognition/models/posenet.npz")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, weight)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    i = 0

    while True:
        ret, img = cap.read()

        if not ret:
            print("Failed to capture image")
            break
        if i == frequency:
            i = 0
            person_pose_array = pose_detector(img, fast_mode=True)
            for i in range(len(person_pose_array)):
                x = person_pose_array[i][0][0]
                y = person_pose_array[i][0][1]
                for j in range(len(person_pose_array[i])):
                    if person_pose_array[i][j][2] != 0:
                        person_pose_array[i][j][0] -= x
                        person_pose_array[i][j][1] -= y
            person_pose_array = np.array(person_pose_array[:][:][0:2])
            path = "result/camera/" + str(time.time()) + ".npy"
            print(path)
            np.save(path, person_pose_array)

            # res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)
            # cv2.imshow("result", res_img)
        cv2.imshow("result", img)
        i += 1
        cv2.waitKey(1)


if __name__ == '__main__':
    main(frequency=5)
