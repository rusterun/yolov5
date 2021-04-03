import argparse
import time
from pathlib import Path
import os
import xmltodict
import cv2
import torch
import torch.backends.cudnn as cudnn
from spam import send_subscribes

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

cams = {
"h1037.mp4":[61.78526847055156,34.34672176837922],
"h1265.mp4":[61.781778495233844,34.35643672943116],
"h1267.mp4":[61.78303148253981,34.3603527545929],
"h1281.mp4":[61.786133294167215,34.3504285812378],
"h2047.mp4":[61.78055589564591,34.3524992465973],
"h2857.mp4":[61.784822100456175,34.35787439346314],
"h3023.mp4":[61.78371375577274,34.353818893432624],
"h4293.mp4":[61.7823365120819,34.33016180992127],
"h1327.mp4":[61.78189517232132,34.31914329528809],
"h1235.mp4":[61.78855010622828,34.35263872146607]
}

places = {
"h1037.mp4":'Шотмана ул. - Ленина пр.',
"h1265.mp4":'Анохина ул. - Гоголя ул.',
"h1267.mp4":'Антикайнена ул. - Гоголя ул.',
"h1281.mp4":'Ленина пр. - Анохина ул.',
"h2047.mp4":'Красноармейская ул. - Гоголя ул',
"h2857.mp4":'Антикайнена ул. - М.Горького ул.',
"h3023.mp4":'Анохина ул. - М.Горького ул.',
"h4293.mp4":'Ватутина ул - 2-я Северная ул.',
"h1327.mp4":'Чапаева ул. - Пархоменко ул.',
"h1235.mp4":'Антикайнена ул. - Красная ул.'
}


def detect(src, save_img=False):
    dist, source, weights, view_img, save_txt, imgsz = opt.waiting, src, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    
    if source.endswith('.mp4'):
        name = source.split('/')[-1]
        coords = cams[name]
    else:
        name = 'This format not поддерживается'
        coords = [0,0]

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = ['fox']
    colors = [[212, 0, 219], [117, 117, 255]]

    is_not_wrote = False #Sarapultsev's var

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()


    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        frame_number = 0
        fox=0

        for i, det in enumerate(pred):  # detections per image
            frame_number+=1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] # normalization gain whwh
            if len(det):

                if len(det) > 1:
                    tmp=[]
                    for ob in det:
                        tmp.append(ob[-2])
                    val = (det[-2] == max(tmp)).nonzero(as_tuple=True)[0]
                    for obi in range(len(det)):
                        if obi != val:
                            det[obi][:4] = torch.cuda.FloatTensor([0, 0, 0, 0])


                if dist > 1 and opt.waiting > 1:
                    if opt.is_view_waiting: #hyperparametr --is_view_waiting

                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} maybe... wait, wait, wait... "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # Add bbox to image
                                label = f'maybe... wait'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)+1], line_thickness=3)
                    else:
                      is_not_wrote = True
                    dist-=1
                elif dist == 1:
                    fox+=1
                    first_frame=frame_number-7
                    
                    

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    else:
                        raise('maybe --waiting < 1, it must have arg > 0. Or ask Sarapultsev')


                else:
                    dist=opt.waiting
                    is_not_wrote = True
                    if fox:
                        send_subscribes(f"Лиса проехала\nГде? {places[name]}\nКак долго она была в кадре? {(frame_number-first_frame)//25} секунд")
                        fox=0

            if is_not_wrote:
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--waiting', type=int, default=1, help='distance from frame to frame или спросите у Вани')
    parser.add_argument('--is_view_waiting', default=False, action='store_true', help='Показывать ли сомнения сети или спросите у Вани')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                if opt.source.endswith('/'):
                    for src in os.listdir(opt.source[:-1]):
                        detect(opt.source+src)
                else:
                    detect(opt.source)
                strip_optimizer(opt.weights)
        else:
            if opt.source.endswith('/'):
                for src in os.listdir(opt.source[:-1]):
                    detect(opt.source+src)
            else:
                detect(opt.source)
    frtmp = dict()
    list_keys = list(frtmp.keys()).sort()
    res_dict = dict()
    num=0
    for i in list_keys:
        num+=1
        res_dict[num] = frtmp[i]
    
    with open('../data.xml', 'w') as f:
        f.write(xmltodict.unparse(res_dict))


    
    

