import cv2
import chainer
import os
import time
import sys
sys.path.append("../../")
from PosturalRecognition.train.pose_detector import PoseDetector, draw_person_pose
from PosturalRecognition.train.face_detector import FaceDetector, draw_face_keypoints
from PosturalRecognition.train.hand_detector import HandDetector, draw_hand_keypoints

chainer.using_config('enable_backprop', False)

# load model
pose_detector = PoseDetector("posenet", "../models/posenet.npz")
hand_detector = HandDetector("handnet", "../models/handnet.npz")
face_detector = FaceDetector("facenet", "../models/facenet.npz")


def get_person_pose_array(img_file, result='../result.png', onlypos=True, save=False, rectangle=False, image=False):
    img = cv2.imread(img_file)

    # print("Estimating pose...")
    person_pose_array = pose_detector(img)
    res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)
    if not onlypos:
        # each person detected
        for person_pose in person_pose_array:
            unit_length = pose_detector.get_unit_length(person_pose)

            # face estimation
            # print("Estimating face keypoints...")
            cropped_face_img, bbox = pose_detector.crop_face(img, person_pose, unit_length)
            if cropped_face_img is not None:
                face_keypoints = face_detector(cropped_face_img)
                res_img = draw_face_keypoints(res_img, face_keypoints, (bbox[0], bbox[1]))
                if rectangle:
                    cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

            # hands estimation
            # print("Estimating hands keypoints...")
            hands = pose_detector.crop_hands(img, person_pose, unit_length)
            if hands["left"] is not None:
                hand_img = hands["left"]["img"]
                bbox = hands["left"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="left")
                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                if rectangle:
                    cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

            if hands["right"] is not None:
                hand_img = hands["right"]["img"]
                bbox = hands["right"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="right")
                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                if rectangle:
                    cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
        # print('Saving result into result.png...')
    if save:
        cv2.imwrite(result, res_img)
        print("save done.")
    if not image:
        return person_pose_array
    else:
        return res_img


def getPoseArray(img):
    # print("pose img: " + str(img))
    person_pose_array = get_person_pose_array(img)
    for i in range(len(person_pose_array)):
        x = person_pose_array[i][0][0]
        y = person_pose_array[i][0][1]
        for j in range(len(person_pose_array[i])):
            if person_pose_array[i][j][2] != 0:
                person_pose_array[i][j][0] -= x
                person_pose_array[i][j][1] -= y
    return person_pose_array


def draw_video(video_path, img_save_path, save_name):
    video = cv2.VideoCapture(video_path)

    if video.isOpened():
        # 判断是否正常打开
        rval, frame = video.read()
    else:
        print("Open video file filed.")
        rval = False
    c = 1
    # 循环读取视频帧
    while rval:
        rval, frame = video.read()
        cv2.imwrite(img_save_path + "/" + str(c) + ".jpg", frame)
        c += 1
        cv2.waitKey(1)
    video.release()

    img_list_posonly = []
    img_list_all = []
    img_list_rec = []

    for (root, dirs, imgfiles) in os.walk(img_save_path):
        for i in range(len(imgfiles)):
            imgfiles[i] = imgfiles[i].split('.')
            imgfiles[i][0] = int(imgfiles[i][0])

        imgfiles.sort()

        for i in range(len(imgfiles)):
            imgfiles[i][0] = str(imgfiles[i][0])
            imgfiles[i] = imgfiles[i][0] + '.' + imgfiles[i][1]

        imgfiles = imgfiles[:-1]

        for i in range(len(imgfiles)):
            print(imgfiles[i], end='\t')
            time_start = time.time()

            img_path = os.path.join(root, imgfiles[i])

            img_posonly = get_person_pose_array(img_file=img_path, result="", onlypos=True, save=False, rectangle=False, image=True)
            img_all = get_person_pose_array(img_file=img_path, result="", onlypos=False, save=False, rectangle=False, image=True)
            img_rec = get_person_pose_array(img_file=img_path, result="", onlypos=False, save=False, rectangle=True, image=True)

            img_list_posonly.append(img_posonly)
            img_list_all.append(img_all)
            img_list_rec.append(img_rec)

            time_end = time.time()
            print(time_end-time_start)

    fps = 30  # 视频帧率
    import cv
    # fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter_posonly = cv2.VideoWriter(save_name + "_pos_only.avi", fourcc, fps, (960, 544))
    videoWriter_all = cv2.VideoWriter(save_name + "_all_points.avi", fourcc, fps, (960, 544))
    videoWriter_rec = cv2.VideoWriter(save_name + "_with_rectangle.avi", fourcc, fps, (960, 544))
    for i in range(len(img_list_posonly)):
        videoWriter_posonly.write(img_list_posonly[i])
        videoWriter_all.write(img_list_all[i])
        videoWriter_rec.write(img_list_rec[i])
    videoWriter_posonly.release()
    videoWriter_all.release()
    videoWriter_rec.release()



if __name__ == '__main__':
    # x = getPoseArray("data/person.png")
    # get_person_pose_array("../../data/1.jpg", "../../data/1_result.png", False, True)
    # get_person_pose_array("../../data/2.png", "../../data/2_result.png", False, True)
    # get_person_pose_array("../../data/3.jpg", "../../data/3_result.png", False, True)
    # get_person_pose_array("../../data/4.jpg", "../../data/4_result.png", False, True)
    # get_person_pose_array("../../data/5.jpg", "../../data/5_result.png", False, True)
    # get_person_pose_array("../../data/7.png", "../../data/7_result.png", False, True)
    # get_person_pose_array("../../data/9.jpg", "../../data/9_result.png", False, True)
    # get_person_pose_array("../../data/person.png", "../../data/person_result.png", False, True)
    # get_person_pose_array("../../data/gta5.jpg", "../../data/gta5_result.png", False, True)
    # draw_video("../../data/v1.mp4", "../../img/tmp", "v1")
    get_person_pose_array("../../data/11.png", "../../data/11_result.png", False, True)
    get_person_pose_array("../../data/12.png", "../../data/12_result.png", False, True)
    get_person_pose_array("../../data/13.png", "../../data/13_result.png", False, True)
    get_person_pose_array("../../data/14.png", "../../data/14_result.png", False, True)

