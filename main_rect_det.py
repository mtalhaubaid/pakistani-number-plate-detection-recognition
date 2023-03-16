import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import torch.backends.cudnn as cudnn
from utils.general import set_logging
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size,
                           check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                           scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
# import pytesseract
import easyocr
# import psycopg2
import time
from datetime import datetime as ddt
from datetime import timezone as tz
from sort import *

# Establishing the connection
# conn = psycopg2.connect(
#    database="Surveillance", user='postgres', password='admin12345', host='127.0.0.1', port= '5432'
# )
# cursor = conn.cursor()
##### DEFINING GLOBAL VARIABLE

EASY_OCR = easyocr.Reader(['en'])  ### initiating easyocr
OCR_TH = 0.05

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

np_counter = 0


def recognize_plate_easyocr(img, coords, reader, region_threshold):
    # separate coordinates from box
    global np_counter
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    nplate = img[int(ymin) - 2:int(ymax) + 2, int(xmin) - 2:int(xmax) + 2]
    # nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]  ### cropping the number plate from the whole image
    cv2.imwrite("croped_np/" + str(np_counter) + "_.jpg", nplate)
    np_counter += 1
    ocr_result = reader.readtext(nplate)
    text = filter_text(region=15000, ocr_result=ocr_result, region_threshold=region_threshold)

    if len(text) == 1:
        text = text[0].upper()
    return text


### to filter out wrong detections

def filter_text(region, ocr_result, region_threshold):
    # rectangle_size = region.shape[0] * region.shape[1]
    rectangle_size=15000
    plate = []
    # print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate


"""Function to Draw Bounding boxes"""


def draw_boxes(img, bbox, plate_num, identities=None, categories=None, names=None, color_box=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        coords = [x1, y1, x2, y2]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
        label = str(id)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 191, 0), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 191, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    [255, 255, 255], 1)
        # cv2.putText(img, plate_num, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 2)
        # cv2.circle(img, data, 3, (255,191,0),-1)
    return img


# ..............................................................................

check_num = []

count_plate=0
@torch.no_grad()
def detect(weights=ROOT / 'yolov5n.pt',
           source=ROOT / 'yolov5/data/images',
           data=ROOT / 'yolov5/data/coco128.yaml',
           imgsz=(1080, 720), conf_thres=0.9, iou_thres=0.9,
           max_det=1000, device='cpu', view_img=False,
           save_txt=False, save_conf=False, save_crop=False,
           nosave=False, classes=None, agnostic_nms=False,
           augment=False, visualize=False, update=False,
           project=ROOT / 'runs/detect', name='exp',
           exist_ok=False, line_thickness=2, hide_labels=False,
           hide_conf=False, half=False, dnn=False, display_labels=False,
           blur_obj=False, color_box=False, ):
    save_img = not nosave and not source.endswith('.txt')

    # .... Initialize SORT ....
    sort_max_age = 5
    sort_min_hits = 4
    sort_iou_thresh = 0.4
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)

    half &= (pt or jit or onnx or engine) and device.type != 'cpu'
    if pt or jit:
        model.model.half() if half else model.model.float()

    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    t0 = time.time()

    dt, seen = [0.0, 0.0, 0.0], 0
    plate_num = 0
    vehicle_count=0
    for path, im, im0s, vid_cap, s in dataset:

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            start_point = (500, 400)

            end_point = (1200, 700)
            cv2.rectangle(im0, start_point, end_point, (255, 0, 0), 2)
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                # Run SORT
                # (245, 158), (721, 924),
                px1, py1, px2, py2 = start_point[0], start_point[1], end_point[0], end_point[1]
                hx1, hy1, hx2, hy2 = dets_to_sort[0][0], dets_to_sort[0][1], dets_to_sort[0][2], dets_to_sort[0][3]
                if hx1 >= px1 and hy1 >= py1 and hx2 <= px2 and hy2 <= py2:
                    # if dets_to_sort[0][2]< start_point[0]:
                    tracked_dets = sort_tracker.update(dets_to_sort)
                    tracks = sort_tracker.getTrackers()

                    # draw boxes for visualization

                    if len(tracked_dets) > 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]

                        for i in range(len(bbox_xyxy)):
                            # start_point[0]
                            # if bbox_xyxy[i][0] < start_point[0]:
                            if identities[i] not in check_num:
                                check_num.append(identities[i])
                                plate_num = recognize_plate_easyocr(img=im0, coords=bbox_xyxy[i], reader=EASY_OCR,
                                                                    region_threshold=OCR_TH)
                                if plate_num:
                                    vehicle_count += 1
                                    print(plate_num)
                                    print("No. of Vehicle",vehicle_count)
                                    dte = ddt.now(tz.utc)
                                else:
                                    count_plate=0

                            draw_boxes(im0, bbox_xyxy, str(plate_num), identities, categories, names, color_box)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        # print("Frame Processing!")
    print("Video Exported Success")



    if update:
        strip_optimizer(weights)

    if vid_cap:
        vid_cap.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='rtsp://admin:admin1234@192.168.1.152:554/cam/realmonitor?channel=1&subtype=0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    detect(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# "rtsp://admin:admin12345@192.168.1.16:554/cam/realmonitor?channel=1&subtype=0"