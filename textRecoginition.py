import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import copy
import numpy as np
import json
import time
import math
import logging
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from tools.infer.utility import (
    draw_ocr_box_txt,
    get_rotate_crop_image,
    get_minarea_rect_crop,
)

# np.set_printoptions(threshold=np.inf)
logger = get_logger()

# args = utility.parse_args()

DET_MODEL_DIR = "./PaddleOCR/models/en_PP-OCRv3_det_infer/"
REC_MODEL_DIR = "./PaddleOCR/models/en_PP-OCRv3_rec_infer/"
E2E_CHAR_DICT_PATH = "./PaddleOCR/ppocr/utils/ic15_dict.txt"
REC_CHAR_DICT_PATH = "./PaddleOCR/ppocr/utils/en_dict.txt"
IMAGE_DIR = "./PaddleOCR/tools/test_img/"
VIS_FONT_PATH = "./doc/fonts/simfang.ttf"
DRAW_IMG_SAVE_DIR = "./inference_results"

class Args():
    def __init__(self):
        self.use_gpu = True
        self.use_xpu = False
        self.use_npu = False
        self.ir_optim = True
        self.use_tensorrt = False
        self.min_subgraph_size = 15
        self.precision = "fp32"
        self.gpu_mem = 500
            # params for text detector
        self.image_dir = IMAGE_DIR
        self.page_num = 0
        self.det_algorithm = 'DB'
        self.det_model_dir = DET_MODEL_DIR
        self.det_limit_side_len = 960
        self.det_limit_type = 'max'
        self.det_box_type = 'quad'
            # DB parmas
        self.det_db_thresh = 0.3
        self.det_db_box_thresh = 0.6
        self.det_db_unclip_ratio = 1.5
        self.max_batch_size = 10
        self.use_dilation = False
        self.det_db_score_mode = "fast"
            # EAST parmas
        self.det_east_score_thresh = 0.8
        self.det_east_cover_thresh = 0.1
        self.det_east_nms_thresh = 0.2
        # SAST parmas
        self.det_sast_score_thresh = 0.5
        self.det_sast_nms_thresh = 0.2
            # PSE parmas
        self.det_pse_thresh = 0
        self.det_pse_box_thresh = 0.85
        self.det_pse_min_area = 16
        self.det_pse_scale = 1
        # FCE parmas
        self.scales = [8, 16, 32]
        self.alpha = 1.0
        self.beta = 1.0
        self.fourier_degree = 5
        # params for text recognizer
        self.rec_algorithm = 'SVTR_LCNet'
        self.rec_model_dir = REC_MODEL_DIR
        self.rec_image_inverse = True
        self.rec_image_shape = "3, 48, 320"
        self.rec_batch_num = 6
        self.max_text_length = 25
        self.rec_char_dict_path = REC_CHAR_DICT_PATH
        self.use_space_char = True
        self.vis_font_path = VIS_FONT_PATH
        self.drop_score = 0.5
        self.e2e_algorithm = 'PGNet'
        self.e2e_model_dir = ""
        self.e2e_limit_side_len = 768
        self.e2e_limit_type = 'max'
        self.e2e_pgnet_score_thresh = 0.5
        self.e2e_char_dict_path = E2E_CHAR_DICT_PATH
        self.e2e_pgnet_valid_set = 'totaltext'
        self.e2e_pgnet_mode = 'fast'
        self.use_angle_cls = False
        self.cls_model_dir = ""
        self.cls_image_shape = "3, 48, 192"
        self.label_list = ['0', '180']
        self.cls_batch_num = 6
        self.cls_thresh = 0.9
        self.enable_mkldnn = False
        self.cpu_threads = 10
        self.use_pdserving = False
        self.warmup = False
        self.sr_model_dir = ""
        self.sr_image_shape = "3, 32, 128"
        self.sr_batch_num = 1
        self.draw_img_save_dir = DRAW_IMG_SAVE_DIR   
        self.save_crop_res = False
        self.crop_res_save_dir = "./output"
        self.use_mp = False
        self.total_process_num = 1
        self.process_id = 0
        self.benchmark = False
        self.save_log_path = "./log_output/"
        self.show_log = True
        self.use_onnx = False

class TextRecoginition(object):
    def __init__(self):
        self.args = Args()
        self.text_detector = predict_det.TextDetector(self.args)
        self.text_recognizer = predict_rec.TextRecognizer(self.args)
        self.use_angle_cls = False
        self.drop_score = self.args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(self.args)

        self.crop_image_res_index = 0

        self.image_file_list = get_image_file_list(self.args.image_dir)
        # self.image_file_list = self.image_file_list[
        #     self.args.process_id :: self.args.total_process_num
        # ]
        # warm up 10 times
        if self.args.warmup:
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = self.__call__(img)

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(
                    output_dir, f"mg_crop_{bno+self.crop_image_res_index}.jpg"
                ),
                img_crop_list[bno],
            )
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {"det": 0, "rec": 0, "csl": 0, "all": 0}
        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict["det"] = elapse
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)
            time_dict["cls"] = elapse

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict["rec"] = elapse
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict["all"] = end - start
        return filter_boxes, filter_rec_res, time_dict

    def pos_process(self, image_input):
        if image_input == "":
            image_input = None
        is_visualize = False
        font_path = self.args.vis_font_path
        drop_score = self.args.drop_score
        # draw_img_save_dir = self.args.draw_img_save_dir
        # os.makedirs(draw_img_save_dir, exist_ok=True)
        save_results = []

        for idx, image_file in enumerate(self.image_file_list):
            flag_gif = False
            flag_pdf = False
            # img, flag_gif, flag_pdf = check_and_read(image_file)
            # if not flag_gif and not flag_pdf:
            #     img = cv2.imread(image_file)
            if image_input is not None:
                img = image_input
            imgs = [img]

            for index, img in enumerate(imgs):
                starttime = time.time()
                dt_boxes, rec_res, time_dict = self.__call__(img)  # call the inference
                elapse = time.time() - starttime
                text_out = ""
                for i in range(len(dt_boxes)):

                    text_out += rec_res[i][0] + " "
                res = [
                    {
                        "transcription": rec_res[i][0],
                        "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                    }
                    for i in range(len(dt_boxes))
                ]
                # if len(imgs) > 1:
                #     save_pred = (
                #         os.path.basename(image_file)
                #         + "_"
                #         + str(index)
                #         + "\t"
                #         + json.dumps(res, ensure_ascii=False)
                #         + "\n"
                #     )
                # else:
                #     save_pred = (
                #         os.path.basename(image_file)
                #         + "\t"
                #         + json.dumps(res, ensure_ascii=False)
                #         + "\n"
                #     )

                min_dist = get_min_distance(res)

                # save_results.append(save_pred)
                image = img #cv2.imread(image_file)
                segmentation_mask = np.zeros(
                    (image.shape[0], image.shape[1]), dtype=np.uint8
                )

                for idx in range(len(res)):
                    poly = np.array(res[idx]["points"], np.int32)
                    # fill polygon in mask
                    segmentation_mask = cv2.fillPoly(segmentation_mask, [poly], 1)

                # cv2.imwrite("binary.jpg", segmentation_mask * 255)
                segmentation_mask = cv2.dilate(
                    segmentation_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, round(min_dist))),
                    iterations=1,
                )
                segmentation_mask = cv2.erode(
                    segmentation_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, round(min_dist))),
                    iterations=1,
                )
                contours, hierarchy = cv2.findContours(
                    image=segmentation_mask,
                    mode=cv2.RETR_TREE,
                    method=cv2.CHAIN_APPROX_NONE,
                )
                image_copy = image.copy()
                box_contour = []
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < 500:
                        cv2.fillPoly(segmentation_mask, pts=[c], color=0)
                        continue

                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    # convert all coordinates floating point values to int
                    box = np.int0(box)
                    box_contour.append(box)
                    cv2.drawContours(image_copy, [box], 0, (0, 255, 0), 1)

                # cv2.imwrite("image_copy.jpg", image_copy)

                if is_visualize:
                    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    boxes = dt_boxes
                    txts = [rec_res[i][0] for i in range(len(rec_res))]
                    scores = [rec_res[i][1] for i in range(len(rec_res))]

                    draw_img = draw_ocr_box_txt(
                        image,
                        boxes,
                        txts,
                        scores,
                        drop_score=drop_score,
                        font_path=font_path,
                    )
                    if flag_gif:
                        save_file = image_file[:-3] + "png"
                    elif flag_pdf:
                        save_file = image_file.replace(
                            ".pdf", "_" + str(index) + ".png"
                        )
                    else:
                        save_file = image_file
                    cv2.imwrite(
                        os.path.join(draw_img_save_dir, os.path.basename(save_file)),
                        draw_img[:, :, ::-1],
                    )

        # with open(
        #     os.path.join(draw_img_save_dir, "system_results.txt"), "w", encoding="utf-8"
        # ) as f:
        #     f.writelines(save_results)
        # print("text_out", text_out)
        return text_out

    def combine_result(self, groupBox, result_list):

        for box_array in groupBox:
            for result in result_list:
                (x_center, y_center) = centroid(result)
                centerPoint = (x_center, y_center)

    def centroid(self, vertexes):
        _x_list = [vertex[0] for vertex in vertexes]
        _y_list = [vertex[1] for vertex in vertexes]
        _len = len(vertexes)
        _x = int(sum(_x_list) / _len)
        _y = int(sum(_y_list) / _len)
        return (_x, _y)

    def is_inside_box(self, bbox, centerPoint):
        condition1 = False
        condition2 = False
        listx = [b[0] for b in bbox]
        listy = [b[1] for b in bbox]
        if centerPoint[0] > min(listx) and centerPoint[0] < max(listx):
            condition1 = True
        if centerPoint[1] > min(listy) and centerPoint[0] < max(listy):
            condition2 = True
        if condition2 == True and condition1 == True:
            return True
        else:
            return False


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, 0, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_min_distance(result_list):
    min_dist = 1e6
    for r in result_list:
        _m = min(
            [
                get_distance(
                    r["points"][i % len(r["points"])],
                    r["points"][(i + 1) % len(r["points"])],
                )
                for i in range(len(r["points"]))
            ]
        )
        min_dist = _m if _m < min_dist else min_dist
    return min_dist


def main():
    image = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
    text_sys = TextRecoginition()
    text_sys.pos_process(image)


if __name__ == "__main__":
    # args = utility.parse_args()
    # if args.use_mp:
    #     p_list = []
    #     total_process_num = args.total_process_num
    #     for process_id in range(total_process_num):
    #         cmd = (
    #             [sys.executable, "-u"]
    #             + sys.argv
    #             + ["--process_id={}".format(process_id), "--use_mp={}".format(False)]
    #         )
    #         p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
    #         p_list.append(p)
    #     for p in p_list:
    #         p.wait()
    # else:
    main()
