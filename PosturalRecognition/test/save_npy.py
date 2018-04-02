import os

from PosturalRecognition.test import VideoCapture
from PosturalRecognition.test import get_person_pose_array
import numpy as np


def save_npy(folder_path, tmp_path, data):
    """
    分帧并获取pose array
    :param folder_path: 视频目录
    :param tmp_path: 帧目录
    :param data: 存储数据名称
    :return:
    """
    result = []
    tmp = []
    for (root, dirs, files) in os.walk(folder_path):
        for j in range(len(files)):
            video_path = os.path.join(root, files[j])
            print(video_path)
            VideoCapture.run(video_path, tmp_path, 2)

            for (imgroot, imgdirs, imgfiles) in os.walk(tmp_path):
                for i in range(len(imgfiles)):
                    imgfiles[i] = imgfiles[i].split('.')
                    imgfiles[i][0] = int(imgfiles[i][0])

                imgfiles.sort()

                for i in range(len(imgfiles)):
                    imgfiles[i][0] = str(imgfiles[i][0])
                    imgfiles[i] = imgfiles[i][0] + '.' + imgfiles[i][1]

                imgfiles = imgfiles[:-1]

                for i in range(len(imgfiles)):
                    img_path = os.path.join(imgroot, imgfiles[i])
                    person_pose_array = get_person_pose_array.getPoseArray(img_path)
                    if i == 0:
                        tmp = [[[[0 for d in range(2)] for c in range(18)] for b in range(len(person_pose_array))] for a
                               in range(len(imgfiles))]
                    for person_pose_index in range(len(person_pose_array)):
                        for joint_index in range(len(person_pose_array[person_pose_index])):
                            try:
                                tmp[i][person_pose_index][joint_index][0] = \
                                person_pose_array[person_pose_index][joint_index][0]
                                tmp[i][person_pose_index][joint_index][1] = \
                                person_pose_array[person_pose_index][joint_index][1]
                            except Exception as e:
                                print(str(files[j]) + "\t" + str(imgfiles[i]) + "\t" + str(i) + "\t" + str(
                                    person_pose_index))
                                print(e)
                tmp = np.array(tmp)
                np.save("result/" + str(j) + ".npy", tmp)
                result.append(tmp)
        result = np.array(result)
        # [视频文件][帧][人物][关节][坐标]
        np.save("data/" + data + ".npy", x)
    return result


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data')
    # args = parser.parse_args()
    data = "jabbing"
    folder_path = "../../data/jabbing"
    # data = str(args.data)
    # folder_path = "KTH/" + data
    tmp_path = "../../img/" + data
    x = save_npy(folder_path, tmp_path, data)
    y = np.load("data/" + data + ".npy")
