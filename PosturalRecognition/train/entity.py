from enum import IntEnum
from PosturalRecognition.models.PoseNet import PoseNet
from PosturalRecognition.models.FaceNet import FaceNet
from PosturalRecognition.models.HandNet import HandNet


class JointType(IntEnum):
    """
    各关节标号
    """
    Nose = 0
    Neck = 1
    RightShoulder = 2
    RightElbow = 3
    RightHand = 4
    LeftShoulder = 5
    LeftElbow = 6
    LeftHand = 7
    RightWaist = 8
    RightKnee = 9
    RightFoot = 10
    LeftWaist = 11
    LeftKnee = 12
    LeftFoot = 13
    RightEye = 14
    LeftEye = 15
    RightEar = 16
    LeftEar = 17


params = {
    'coco_dir': '../../datasets/coco',
    'archs': {
        'posenet': PoseNet,
        'facenet': FaceNet,
        'handnet': HandNet,
    },
    'paf_sigma': 1.3,
    'heatmap_sigma': 1.5,
    'crop_iob_thresh': 0.4,
    'crop_size': 480,
    'input_size': 368,
    'downscale': 8,
    'inference_img_size': 368,
    'inference_scales': [1.0],
    'heatmap_size': 320,
    'gaussian_sigma': 2.5,
    'ksize': 17,
    'n_integ_points': 10,
    'n_integ_points_thresh': 8,
    'heatmap_peak_thresh': 0.1,
    'inner_product_thresh': 0.05,
    'length_penalty_ratio': 0.5,
    'n_subset_limbs_thresh': 7,
    'subset_score_thresh': 0.4,
    'limbs_point': [
        [JointType.Neck, JointType.RightWaist],
        [JointType.RightWaist, JointType.RightKnee],
        [JointType.RightKnee, JointType.RightFoot],
        [JointType.Neck, JointType.LeftWaist],
        [JointType.LeftWaist, JointType.LeftKnee],
        [JointType.LeftKnee, JointType.LeftFoot],
        [JointType.Neck, JointType.RightShoulder],
        [JointType.RightShoulder, JointType.RightElbow],
        [JointType.RightElbow, JointType.RightHand],
        [JointType.RightShoulder, JointType.RightEar],
        [JointType.Neck, JointType.LeftShoulder],
        [JointType.LeftShoulder, JointType.LeftElbow],
        [JointType.LeftElbow, JointType.LeftHand],
        [JointType.LeftShoulder, JointType.LeftEar],
        [JointType.Neck, JointType.Nose],
        [JointType.Nose, JointType.RightEye],
        [JointType.Nose, JointType.LeftEye],
        [JointType.RightEye, JointType.RightEar],
        [JointType.LeftEye, JointType.LeftEar]
    ],
    'coco_joint_indices': [
        JointType.Nose,
        JointType.LeftEye,
        JointType.RightEye,
        JointType.LeftEar,
        JointType.RightEar,
        JointType.LeftShoulder,
        JointType.RightShoulder,
        JointType.LeftElbow,
        JointType.RightElbow,
        JointType.LeftHand,
        JointType.RightHand,
        JointType.LeftWaist,
        JointType.RightWaist,
        JointType.LeftKnee,
        JointType.RightKnee,
        JointType.LeftFoot,
        JointType.RightFoot
    ],

    # 面部参数
    'face_inference_img_size': 368,
    'face_heatmap_peak_thresh': 0.1,
    'face_crop_scale': 1.5,
    'face_line_indices': [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13],
        [13, 14], [14, 15], [15, 16],  # 轮廓
        [17, 18], [18, 19], [19, 20], [20, 21],  # 右眉
        [22, 23], [23, 24], [24, 25], [25, 26],  # 左眉
        [27, 28], [28, 29], [29, 30],  # 鼻子
        [31, 32], [32, 33], [33, 34], [34, 35],  # 鼻子下的横线
        [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],  # 右眼
        [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],  # 左眼
        [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59],
        [59, 48],  # 外唇轮廓
        [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]  # 唇内
    ],

    # 手部参数
    'hand_inference_img_size': 368,
    'hand_heatmap_peak_thresh': 0.1,
    'fingers_indices': [
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        [[0, 5], [5, 6], [6, 7], [7, 8]],
        [[0, 9], [9, 10], [10, 11], [11, 12]],
        [[0, 13], [13, 14], [14, 15], [15, 16]],
        [[0, 17], [17, 18], [18, 19], [19, 20]],
    ],
}
