import os
import sys
import subprocess
import cv2
import json
import time
import logging
from copy import deepcopy
from paddleocr.ppocr.utils.utility import get_image_file_list, check_and_read
from paddleocr.ppocr.utils.logging import get_logger
from paddleocr.ppstructure.table.predict_table import TableSystem, to_excel
from paddleocr.ppstructure.utility import parse_args, draw_structure_result, cal_ocr_word_box

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

logger = get_logger()


class StructureSystem(object):
    def __init__(self, args):
        self.mode = args.mode

        if self.mode == "structure":
            if not args.show_log:
                logger.setLevel(logging.INFO)
            if args.layout is False and args.ocr is True:
                args.ocr = False
                logger.warning(
                    "When args.layout is false, args.ocr is automatically set to false"
                )
            # init model
            self.text_system = None
            self.table_system = None

            if args.table:
                if self.text_system is not None:
                    self.table_system = TableSystem(
                        args,
                        self.text_system.text_detector,
                        self.text_system.text_recognizer,
                    )
                else:
                    self.table_system = TableSystem(args)
        self.return_word_box = args.return_word_box

    def __call__(self, img, layout_res, return_ocr_result_in_table=False, img_idx=0):
        time_dict = {
            "layout": 0,
            "table": 0,
            "table_match": 0,
            "det": 0,
            "rec": 0,
            "kie": 0,
            "all": 0,
        }
        start = time.time()

        if self.mode == "structure":
            ori_im = img.copy()

            if layout_res is not None:
                layout_res = layout_res
            else:
                h, w = ori_im.shape[:2]
                layout_res = [dict(bbox=None, label="table", score=0.0)]

            # As reported in issues such as #10270 and #11665, the old
            # implementation, which recognizes texts from the layout regions,
            # has problems with OCR recognition accuracy.
            #
            # To enhance the OCR recognition accuracy, we implement a patch fix
            # that first use text_system to detect and recognize all text information
            # and then filter out relevant texts according to the layout regions.
            text_res = None
            if self.text_system is not None:
                text_res, ocr_time_dict = self._predict_text(img)
                time_dict["det"] += ocr_time_dict["det"]
                time_dict["rec"] += ocr_time_dict["rec"]

            res_list = []
            for region in layout_res:
                res = ""
                if region["bbox"] is not None:
                    x1, y1, x2, y2 = region["bbox"]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im
                bbox = [x1, y1, x2, y2]

                if region["type"] == "table":
                    if self.table_system is not None:
                        res, table_time_dict = self.table_system(
                            roi_img, return_ocr_result_in_table
                        )
                        time_dict["table"] += table_time_dict["table"]
                        time_dict["table_match"] += table_time_dict["match"]
                        time_dict["det"] += table_time_dict["det"]
                        time_dict["rec"] += table_time_dict["rec"]

                else:
                    if text_res is not None:
                        # Filter the text results whose regions intersect with the current layout bbox.
                        res = self._filter_text_res(text_res, bbox)

                res_list.append(
                    {
                        "type": region["type"].lower(),
                        "bbox": bbox,
                        "img": roi_img,
                        "res": res,
                        "img_idx": img_idx,
                        "score": region["score"],
                    }
                )

            end = time.time()
            time_dict["all"] = end - start
            return res_list, time_dict
        return None, None

    def _predict_text(self, img):
        filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(img)

        # remove style char,
        # when using the recognition model trained on the PubtabNet dataset,
        # it will recognize the text format in the table, such as <b>
        style_token = [
            "<strike>",
            "<strike>",
            "<sup>",
            "</sub>",
            "<b>",
            "</b>",
            "<sub>",
            "</sup>",
            "<overline>",
            "</overline>",
            "<underline>",
            "</underline>",
            "<i>",
            "</i>",
        ]
        res = []
        for box, rec_res in zip(filter_boxes, filter_rec_res):
            rec_str, rec_conf = rec_res[0], rec_res[1]
            for token in style_token:
                if token in rec_str:
                    rec_str = rec_str.replace(token, "")
            if self.return_word_box:
                word_box_content_list, word_box_list = cal_ocr_word_box(
                    rec_str, box, rec_res[2]
                )
                res.append(
                    {
                        "text": rec_str,
                        "confidence": float(rec_conf),
                        "text_region": box.tolist(),
                        "text_word": word_box_content_list,
                        "text_word_region": word_box_list,
                    }
                )
            else:
                res.append(
                    {
                        "text": rec_str,
                        "confidence": float(rec_conf),
                        "text_region": box.tolist(),
                    }
                )
        return res, ocr_time_dict

    def _filter_text_res(self, text_res, bbox):
        res = []
        for r in text_res:
            box = r["text_region"]
            rect = box[0][0], box[0][1], box[2][0], box[2][1]
            if self._has_intersection(bbox, rect):
                res.append(r)
        return res

    def _has_intersection(self, rect1, rect2):
        x_min1, y_min1, x_max1, y_max1 = rect1
        x_min2, y_min2, x_max2, y_max2 = rect2
        if x_min1 > x_max2 or x_max1 < x_min2:
            return False
        if y_min1 > y_max2 or y_max1 < y_min2:
            return False
        return True


def load_structure_res(output_folder, img_name, img_idx=0):
    save_folder = os.path.join(args.output, "structure")
    # Construct the path to the .txt file
    res_file_path = os.path.join(save_folder, img_name, f"res_{img_idx}.txt")

    if not os.path.exists(res_file_path):
        raise FileNotFoundError(f"The file {res_file_path} does not exist.")

    # Read and load the content
    results = []
    with open(res_file_path, 'r', encoding='utf8') as f:
        for line in f:
            region = json.loads(line.strip())
            results.append(region)

    return results


def save_structure_res(res, save_folder, img_name, img_idx=0):
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    res_cp = deepcopy(res)
    # save res
    with open(
        os.path.join(excel_save_folder, "res_{}.txt".format(img_idx)),
        "w",
        encoding="utf8",
    ) as f:
        for region in res_cp:
            roi_img = region.pop("img")
            f.write("{}\n".format(json.dumps(region)))

            if (
                region["type"].lower() == "table"
                and len(region["res"]) > 0
                and "html" in region["res"]
            ):
                excel_path = os.path.join(
                    excel_save_folder, "{}_{}.xlsx".format(region["bbox"], img_idx)
                )
                to_excel(region["res"]["html"], excel_path)
            elif region["type"].lower() == "figure":
                img_path = os.path.join(
                    excel_save_folder, "{}_{}.jpg".format(region["bbox"], img_idx)
                )
                cv2.imwrite(img_path, roi_img)


def main(args, layout_res):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list
    image_file_list = image_file_list[args.process_id::args.total_process_num]

    if not args.use_pdf2docx_api:
        structure_sys = StructureSystem(args)
        save_folder = os.path.join(args.output, structure_sys.mode)
        os.makedirs(save_folder, exist_ok=True)
    img_num = len(image_file_list)

    for i, image_file in enumerate(image_file_list):
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        img, flag_gif, flag_pdf = check_and_read(image_file)
        img_name = os.path.basename(image_file).split(".")[0]

        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)

        if not flag_pdf:
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            imgs = img

        for index, img in enumerate(imgs):
            res, time_dict = structure_sys(img, layout_res, img_idx=index)
            img_save_path = os.path.join(
                save_folder, img_name, "show_{}.jpg".format(index)
            )
            os.makedirs(os.path.join(save_folder, img_name), exist_ok=True)
            if structure_sys.mode == "structure" and res != []:
                draw_img = draw_structure_result(img, res, font_path=args.vis_font_path)
                save_structure_res(res, save_folder, img_name, index)

            if res != []:
                cv2.imwrite(img_save_path, draw_img)
                logger.info("result save to {}".format(img_save_path))

        logger.info("Predict time : {:.3f}s".format(time_dict["all"]))


if __name__ == "__main__":
    args = parse_args()
    args.use_gpu = False
    args.image_dir = r"C:\Users\dimitrir.nguinwam\Downloads\img"

    args.det_model_dir = "inference_table/en_PP-OCRv3_det_infer"
    args.rec_model_dir = "inference_table/en_PP-OCRv3_rec_infer"
    args.table_model_dir = "inference_table/model_final"
    args.rec_char_dict_path = "dict_table/en_dict.txt"
    args.table_char_dict_path = "dict_table/table_structure_dict.txt"
    args.font_path = r'C:\Windows\Fonts\Arial.ttf'

    args.output = "../output"
    args.layout = False
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = (
                [sys.executable, "-u"]
                + sys.argv
                + ["--process_id={}".format(process_id), "--use_mp={}".format(False)]
            )
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
