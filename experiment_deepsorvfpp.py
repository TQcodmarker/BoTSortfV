import argparse
import ast
import os
import re
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
BOT_SORT_ROOT = os.path.join(ROOT, 'BoT-SORT')
if BOT_SORT_ROOT not in sys.path:
    sys.path.insert(0, BOT_SORT_ROOT)

from tracker.bot_sort import BoTSORT
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess


def parse_clip_time(video_path):
    parts = re.split(r'[\.\-\_\\/]', video_path)
    return [
        int(parts[-11]), int(parts[-10]), int(parts[-9]),
        int(parts[-8]), int(parts[-7]), int(parts[-6]), 0
    ]


def time_to_stamp_ms(time_value):
    name = "%d_%02d_%02d_%02d_%02d_%02d_%03d" % tuple(time_value)
    dt = datetime.strptime(name, "%Y_%m_%d_%H_%M_%S_%f")
    stamp = int(time.mktime(dt.timetuple()) * 1000.0 + dt.microsecond / 1000.0)
    return stamp, name


def update_time_ms(time_value, step_ms):
    time_value[6] += step_ms
    if time_value[6] >= 1000:
        time_value[5] += 1
        time_value[6] -= 1000
        if time_value[5] >= 60:
            time_value[4] += 1
            time_value[5] -= 60
            if time_value[4] >= 60:
                time_value[3] += 1
                time_value[4] -= 60
    return time_to_stamp_ms(time_value)


def read_camera_para(path):
    with open(path, 'r') as f:
        return list(map(float, ast.literal_eval(f.readline().strip())))


def find_clip_video(data_path):
    for name in os.listdir(data_path):
        lower = name.lower()
        if lower.endswith('.mp4') or lower.endswith('.avi'):
            return os.path.join(data_path, name)
    raise FileNotFoundError('No video file found under {}'.format(data_path))


def apply_ablation_mode(args):
    if args.mode == 'botsort':
        args.ais_path = None
        args.camera_para = None
        args.ais_cost_weight = 0.0
        args.ais_heading_weight = 0.0
    elif args.mode == 'virtual':
        args.ais_cost_weight = 0.0
        args.ais_heading_weight = 0.0
    elif args.mode == 'assoc':
        args.ais_heading_weight = 0.0
    elif args.mode == 'full':
        pass
    else:
        raise ValueError('Unsupported mode: {}'.format(args.mode))


class Predictor(object):
    def __init__(self, model, exp, device, fp16=False):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {
            'raw_img': img,
            'height': img.shape[0],
            'width': img.shape[1],
        }
        proc_img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info['ratio'] = ratio
        proc_img = torch.from_numpy(proc_img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            proc_img = proc_img.half()
        with torch.no_grad():
            outputs = self.model(proc_img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


def build_predictor(args):
    if args.exp_file is None:
        raise ValueError('Please pass --exp-file, e.g. BoT-SORT/yolox/exps/example/mot/yolox_x_mix_det.py')
    exp = get_exp(args.exp_file, args.name)
    if args.conf is not None:
        exp.test_conf = args.conf
    else:
        exp.test_conf = max(0.001, args.track_low_thresh - 0.01)
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    model = exp.get_model().to(device)
    print('Model Summary:', get_model_info(model, exp.test_size))
    model.eval()

    if args.ckpt is None:
        raise ValueError('Please pass --ckpt for the YOLOX detector checkpoint.')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model = model.half()
    return Predictor(model, exp, device, args.fp16)


def write_result_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.writelines(lines)


def run_experiment(args):
    apply_ablation_mode(args)

    video_path = args.video_path or find_clip_video(args.data_path)
    ais_path = args.ais_path
    camera_para_path = args.camera_para_path or os.path.join(args.data_path, 'camera_para.txt')
    if args.camera_para is None and camera_para_path and os.path.exists(camera_para_path):
        args.camera_para = read_camera_para(camera_para_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Unable to open video: {}'.format(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = args.fps
    args.fps = int(round(fps))
    frame_step_ms = int(1000 / args.fps)
    initial_time = parse_clip_time(video_path)
    timestamp0, time_name0 = time_to_stamp_ms(initial_time)

    print('Video:', video_path)
    print('AIS:', ais_path)
    print('Start time:', time_name0, 'timestamp:', timestamp0, 'fps:', args.fps)
    print('Mode:', args.mode)

    predictor = build_predictor(args)
    tracker = BoTSORT(args, frame_rate=args.fps)

    tracking_lines = []
    fusion_lines = []
    frame_id = 0
    total_det_track_time = 0.0
    total_track_time = 0.0

    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out_video_path = os.path.join(args.output_dir, args.name + '_' + args.mode + '.mp4')
        writer = None
    else:
        writer = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_id += 1
        timestamp, time_name = update_time_ms(initial_time, frame_step_ms)

        start = time.time()
        outputs, img_info = predictor.inference(frame)
        scale = min(
            predictor.test_size[0] / float(img_info['height']),
            predictor.test_size[1] / float(img_info['width'])
        )
        if outputs[0] is not None:
            detections = outputs[0].cpu().numpy()[:, :7]
            detections[:, :4] /= scale
        else:
            detections = np.empty((0, 7), dtype=float)

        track_start = time.time()
        online_targets = tracker.update(detections, img_info['raw_img'], timestamp=timestamp)
        total_track_time += time.time() - track_start
        total_det_track_time += time.time() - start

        for target in online_targets:
            tlwh = target.tlwh
            tid = target.track_id
            vertical = tlwh[2] / max(tlwh[3], 1e-6) > args.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] <= args.min_box_area or vertical:
                continue

            tracking_lines.append(
                '{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1\n'.format(
                    frame_id, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], target.score
                )
            )
            if getattr(target, 'ais_id', None) is not None:
                fusion_lines.append(
                    '{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1\n'.format(
                        frame_id, target.ais_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], target.score
                    )
                )

            if args.save_video:
                x, y, w, h = [int(v) for v in tlwh]
                color = (0, 255, 0) if getattr(target, 'ais_id', None) is not None else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, str(tid), (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if args.save_video:
            if writer is None:
                writer = cv2.VideoWriter(out_video_path, fourcc, args.fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)

        if frame_id % args.log_interval == 0:
            print('Frame {:05d} time={} avg_e2e={:.4f}s avg_track={:.4f}s'.format(
                frame_id, time_name,
                total_det_track_time / frame_id,
                total_track_time / frame_id
            ))

        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    result_prefix = os.path.join(args.output_dir, args.name + '_' + args.mode)
    write_result_lines(result_prefix + '_tracking.txt', tracking_lines)
    write_result_lines(result_prefix + '_fusion.txt', fusion_lines)
    print('Saved:', result_prefix + '_tracking.txt')
    print('Saved:', result_prefix + '_fusion.txt')
    if frame_id > 0:
        print('Average end-to-end seconds/frame:', total_det_track_time / frame_id)
        print('Average tracker seconds/frame:', total_track_time / frame_id)


def make_parser():
    parser = argparse.ArgumentParser('DeepSORVF++ clip experiment')
    parser.add_argument('--data-path', default='clip-01', help='FVessel clip directory')
    parser.add_argument('--video-path', default=None, help='Override clip video path')
    parser.add_argument('--output-dir', default='result_deepsorvfpp', help='Output directory')
    parser.add_argument('--name', default='clip-01', help='Experiment name')
    parser.add_argument('--mode', default='full', choices=['botsort', 'virtual', 'assoc', 'full'])
    parser.add_argument('--max-frames', type=int, default=-1, help='Stop early for debugging')
    parser.add_argument('--log-interval', type=int, default=30)
    parser.add_argument('--save-video', action='store_true')

    parser.add_argument('-f', '--exp-file', default=None, type=str)
    parser.add_argument('-c', '--ckpt', default=None, type=str)
    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--conf', default=None, type=float)
    parser.add_argument('--nms', default=None, type=float)
    parser.add_argument('--tsize', default=None, type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fuse', action='store_true')

    parser.add_argument('--track_high_thresh', type=float, default=0.6)
    parser.add_argument('--track_low_thresh', type=float, default=0.1)
    parser.add_argument('--new_track_thresh', type=float, default=0.7)
    parser.add_argument('--track_buffer', type=int, default=30)
    parser.add_argument('--match_thresh', type=float, default=0.8)
    parser.add_argument('--aspect_ratio_thresh', type=float, default=10.0)
    parser.add_argument('--min_box_area', type=float, default=10)
    parser.add_argument('--mot20', action='store_true')

    parser.add_argument('--cmc-method', default='none', type=str)
    parser.add_argument('--ablation', action='store_true')

    parser.add_argument('--with-reid', dest='with_reid', action='store_true')
    parser.add_argument('--fast-reid-config', dest='fast_reid_config',
                        default='BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml')
    parser.add_argument('--fast-reid-weights', dest='fast_reid_weights',
                        default='BoT-SORT/pretrained/mot17_sbs_S50.pth')
    parser.add_argument('--proximity_thresh', type=float, default=0.5)
    parser.add_argument('--appearance_thresh', type=float, default=0.25)

    parser.add_argument('--ais-path', default=None)
    parser.add_argument('--camera-para-path', default=None)
    parser.set_defaults(camera_para=None)
    parser.add_argument('--ais-max-age', dest='ais_max_age', type=float, default=2.0)
    parser.add_argument('--ais-kappa', dest='ais_kappa', type=float, default=0.5)
    parser.add_argument('--ais-position-var', dest='ais_position_var', type=float, default=4.0)
    parser.add_argument('--ais-scale-var', dest='ais_scale_var', type=float, default=1000000.0)
    parser.add_argument('--ais-bind-distance', dest='ais_bind_distance', type=float, default=120.0)
    parser.add_argument('--ais-cost-weight', dest='ais_cost_weight', type=float, default=0.25)
    parser.add_argument('--ais-heading-weight', dest='ais_heading_weight', type=float, default=0.05)
    parser.add_argument('--ais-occlusion-min-score', dest='ais_occlusion_min_score', type=float, default=0.4)
    parser.add_argument('--ais-occlusion-max-frames', dest='ais_occlusion_max_frames', type=int, default=60)

    parser.add_argument('--fps', type=int, default=30)
    return parser


if __name__ == '__main__':
    parsed = make_parser().parse_args()
    if parsed.ais_path is None and parsed.mode != 'botsort':
        parsed.ais_path = os.path.join(parsed.data_path, 'ais')
    run_experiment(parsed)
