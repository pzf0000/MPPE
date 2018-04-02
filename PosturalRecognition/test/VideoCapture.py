import cv2
import os


def videocapture(video_path, img_save_path, timeF=1):
    print("vidio file: " + str(video_path))
    print("image save path: " + str(img_save_path))
    print("frame rate: " + str(timeF))
    # 读入视频文件
    video = cv2.VideoCapture(video_path)
    c = 1

    if video.isOpened():
        # 判断是否正常打开
        rval, frame = video.read()
    else:
        print("Open video file filed.")
        rval = False

    # 循环读取视频帧
    while rval:
        rval, frame = video.read()
        # 每隔timeF帧进行存储操作
        if c % timeF == 0:
            # 存储为图像
            cv2.imwrite(img_save_path + "\\" + str(c) + ".jpg", frame)
        c = c + 1
        cv2.waitKey(1)
    video.release()
    print("Frame storage for pictures completed.")


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            try:
                os.remove(path_file)
            except:
                print(str(path_file) + " is being used. cannot delete. pass.")
        else:
            del_file(path_file)


def run(video_path, img_save_path, timeF=1):
    """
    分帧
    :param video_path: 视频路径
    :param img_save_path: 保存路径
    :param timeF: 每多少帧保存一次
    :return:
    """
    if os.path.getsize(img_save_path):
        print("image save path is not empty.")
        del_file(img_save_path)
        print("now, delete all files of the folder.")
    videocapture(video_path, img_save_path, timeF)


if __name__ == '__main__':
    from PosturalRecognition.test import get_person_pose_array
    import numpy as np
    file_name = "person05_walking_d1_uncomp"
    video_path = "datasets/KTH/walking/" + file_name + ".avi"
    img_save_path = "img"
    timeF = 2
    run(video_path, img_save_path, timeF)
    for (imgroot, imgdirs, imgfiles) in os.walk(img_save_path):
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
            print(img_path)
            person_pose_array = get_person_pose_array.getPoseArray(img_path)
            # print("person_pose_array=" + str(len(person_pose_array)))
            # print("imgfiles=" + str(len(imgfiles)))
            if i == 0:
                tmp = [[[[0 for d in range(2)] for c in range(18)] for b in range(1)] for a in range(len(imgfiles))]
            for person_pose_index in range(len(person_pose_array)):
                for joint_index in range(18):
                    try:
                        tmp[i][0][joint_index][0] = person_pose_array[person_pose_index,joint_index,0]
                        tmp[i][0][joint_index][1] = person_pose_array[person_pose_index,joint_index,1]
                    except Exception as e:
                        print(str(imgfiles[i]) + "\t" + str(i) + "\t" + str(person_pose_index))
                        print("1")
                        try:
                            print(tmp[i][person_pose_index][joint_index])
                            print(len(tmp[i][person_pose_index][joint_index]))
                        except:
                            print("2")
                            try:
                                print(tmp[i][person_pose_index])
                                print(len(tmp[i][person_pose_index]))
                            except:
                                print("3")
                                try:
                                    print(tmp[i])
                                    print(len(tmp[i]))
                                except:
                                    print("4")

        tmp = np.array(tmp)
        np.save("result/" + file_name + ".npy", tmp)