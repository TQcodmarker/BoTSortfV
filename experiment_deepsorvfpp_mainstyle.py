import argparse
import os
import sys
import time

import cv2
import imutils
import numpy as np
import pandas as pd
import torch

from utils.file_read import read_all, ais_initial, update_time
from utils.AIS_utils import AISPRO
from utils.draw import DRAW


ROOT = os.path.dirname(os.path.abspath(__file__))
BOT_SORT_ROOT = os.path.join(ROOT, 'BoT-SORT')
if BOT_SORT_ROOT not in sys.path:
    sys.path.insert(0, BOT_SORT_ROOT)

from tracker.bot_sort import BoTSORT
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess


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
        proc_img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        proc_img = torch.from_numpy(proc_img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            proc_img = proc_img.half()

        with torch.no_grad():
            outputs = self.model(proc_img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, ratio


def build_predictor(args):
    if args.exp_file is None:
        raise ValueError('Please pass --exp-file for YOLOX.')
    if args.ckpt is None:
        raise ValueError('Please pass --ckpt for YOLOX.')

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

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model = model.half()
    return Predictor(model, exp, device, args.fp16)


def timestamp_to_ms(value):
    value = float(value)
    if value > 10000000000:
        return value
    return value * 1000.0


def ais_vis_to_records(AIS_vis, frame_timestamp_ms, max_age_sec):
    records = []
    if AIS_vis is None or len(AIS_vis) == 0:
        return records

    for _, row in AIS_vis.iterrows():
        if 'x' not in row or 'y' not in row or pd.isna(row['x']) or pd.isna(row['y']):
            continue
        ais_timestamp_ms = timestamp_to_ms(row['timestamp'])
        delta_t = abs(frame_timestamp_ms - ais_timestamp_ms) / 1000.0
        if delta_t > max_age_sec:
            continue
        records.append({
            'ais_id': row['mmsi'],
            'x': row['x'],
            'y': row['y'],
            'timestamp': ais_timestamp_ms,
            'speed': row.get('speed', None),
            'course': row.get('course', None),
            'heading': row.get('heading', None),
            'lon': row.get('lon', None),
            'lat': row.get('lat', None),
            'reliability': 1.0,
        })
    return records


def empty_ais_dataframe():
    return pd.DataFrame(columns=[
        'mmsi', 'lon', 'lat', 'speed', 'course', 'heading',
        'type', 'x', 'y', 'timestamp'
    ])


def apply_mode(args):
    if args.mode == 'botsort':
        args.ais_cost_weight = 0.0
        args.ais_heading_weight = 0.0
        args.ais_path = None
        args.camera_para = None
    elif args.mode == 'virtual':
        args.ais_cost_weight = 0.0
        args.ais_heading_weight = 0.0
    elif args.mode == 'assoc':
        args.ais_heading_weight = 0.0
    elif args.mode == 'full':
        pass
    else:
        raise ValueError('Unsupported mode: {}'.format(args.mode))


def write_line(path, line):
    with open(path, 'a') as f:
        f.write(line)


def clear_result_files(result_metric):
    for suffix in ['_detection', '_tracking', '_fusion']:
        path = result_metric[:-4] + suffix + result_metric[-4:]
        if os.path.exists(path):
            os.remove(path)


def write_detection_results(result_metric, frame_id, detections):
    path = result_metric[:-4] + '_detection' + result_metric[-4:]
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        score = det[4] if len(det) > 4 else 1.0
        w, h = x2 - x1, y2 - y1
        write_line(path, '{},0,{:.2f},{:.2f},{:.2f},{:.2f},{:.3f},1,1,1\n'.format(
            frame_id, x1, y1, w, h, score))


def write_track_results(result_metric, frame_id, online_targets):
    tracking_path = result_metric[:-4] + '_tracking' + result_metric[-4:]
    fusion_path = result_metric[:-4] + '_fusion' + result_metric[-4:]
    for target in online_targets:
        tlwh = target.tlwh
        line = '{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.3f},1,1,1\n'.format(
            frame_id, target.track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], target.score)
        write_line(tracking_path, line)
        if getattr(target, 'ais_id', None) is not None:
            fusion_line = '{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.3f},1,1,1\n'.format(
                frame_id, target.ais_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], target.score)
            write_line(fusion_path, fusion_line)


def targets_to_deepsorvf_frames(online_targets, AIS_vis, timestamp):
    vis_rows = []
    fus_rows = []
    ais_latest = {}
    if AIS_vis is not None and len(AIS_vis) > 0:
        for mmsi in AIS_vis['mmsi'].unique():
            rows = AIS_vis[AIS_vis['mmsi'] == mmsi].reset_index(drop=True)
            if len(rows) > 0:
                ais_latest[int(mmsi)] = rows.iloc[-1]

    for target in online_targets:
        tlwh = target.tlwh
        x1 = int(max(tlwh[0], 0))
        y1 = int(max(tlwh[1], 0))
        x2 = int(max(tlwh[0] + tlwh[2], 0))
        y2 = int(max(tlwh[1] + tlwh[3], 0))
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        track_id = int(target.track_id)
        ts_sec = int(timestamp // 1000)
        vis_rows.append({
            'ID': track_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'x': cx, 'y': cy, 'timestamp': ts_sec
        })

        ais_id = getattr(target, 'ais_id', None)
        if ais_id is None:
            continue
        try:
            ais_id_int = int(ais_id)
        except (TypeError, ValueError):
            continue
        if ais_id_int not in ais_latest:
            continue
        ais = ais_latest[ais_id_int]
        fus_rows.append({
            'ID': track_id,
            'mmsi': ais_id_int,
            'lon': ais['lon'],
            'lat': ais['lat'],
            'speed': ais['speed'],
            'course': ais['course'],
            'heading': ais['heading'],
            'type': ais['type'],
            'x1': x1,
            'y1': y1,
            'w': int(tlwh[2]),
            'h': int(tlwh[3]),
            'timestamp': int(ais['timestamp'])
        })

    Vis_cur = pd.DataFrame(vis_rows, columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'timestamp'])
    Fus_tra = pd.DataFrame(fus_rows, columns=[
        'ID', 'mmsi', 'lon', 'lat', 'speed', 'course', 'heading', 'type',
        'x1', 'y1', 'w', 'h', 'timestamp'
    ])
    return Vis_cur, Fus_tra


def run(args):
    video_path, ais_path, result_video, result_metric, initial_time, camera_para = read_all(
        args.data_path, args.result_path)
    clear_result_files(result_metric)

    args.ais_path = ais_path
    args.camera_para = camera_para
    apply_mode(args)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Unable to open video: {}'.format(video_path))

    im_shape = [cap.get(3), cap.get(4)]
    fps = int(cap.get(5))
    if fps <= 0:
        fps = args.fps
    args.fps = fps
    t = int(1000 / fps)

    ais_file, timestamp0, time0 = ais_initial(ais_path, initial_time)
    Time = initial_time.copy()

    AIS = AISPRO(ais_path, ais_file, im_shape, t)
    DRA = DRAW(im_shape, t)
    predictor = build_predictor(args)
    tracker = BoTSORT(args, frame_rate=fps)

    os.makedirs(os.path.dirname(result_video), exist_ok=True)
    writer = None
    Vis_tra = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'timestamp'])
    frame_id = 0
    total_time = []

    print('Start Time: {} || Stamp: {} || fps: {} || mode: {}'.format(
        time0, timestamp0, fps, args.mode))

    while True:
        ok, im = cap.read()
        if not ok or im is None:
            break
        frame_id += 1
        start = time.time()

        Time, timestamp, Time_name = update_time(Time, t)

        if args.mode == 'botsort':
            AIS_vis = empty_ais_dataframe()
            AIS_cur = empty_ais_dataframe()
            ais_frame = []
        else:
            AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)
            ais_frame = ais_vis_to_records(AIS_vis, timestamp, args.ais_max_age)

        outputs, ratio = predictor.inference(im)
        if outputs[0] is not None:
            detections = outputs[0].cpu().numpy()[:, :7]
            detections[:, :4] /= ratio
        else:
            detections = np.empty((0, 7), dtype=float)

        online_targets = tracker.update(
            detections, im, ais_frame=ais_frame, timestamp=timestamp)

        filtered_targets = []
        for target in online_targets:
            tlwh = target.tlwh
            vertical = tlwh[2] / max(tlwh[3], 1e-6) > args.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                filtered_targets.append(target)

        write_detection_results(result_metric, frame_id, detections)
        write_track_results(result_metric, frame_id, filtered_targets)

        Vis_cur, Fus_tra = targets_to_deepsorvf_frames(filtered_targets, AIS_vis, timestamp)
        if len(Vis_cur) > 0:
            Vis_tra = pd.concat([Vis_tra, Vis_cur], ignore_index=True)
            Vis_tra = Vis_tra.drop(Vis_tra[Vis_tra['timestamp'] < (timestamp // 1000 - 2 * 60)].index)
        result = DRA.draw_traj(im, AIS_vis, AIS_cur, Vis_tra, Vis_cur, Fus_tra, timestamp)
        result = imutils.resize(result, height=args.show_size)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(result_video[:-4] + '_' + args.mode + result_video[-4:],
                                     fourcc, fps, (result.shape[1], result.shape[0]))
        writer.write(result)

        elapsed = time.time() - start
        total_time.append(elapsed)
        if timestamp % 1000 < t or frame_id % args.log_interval == 0:
            print('Time: {} || Frame: {} || Stamp: {} || Process: {:.6f} || Average: {:.6f}'.format(
                Time_name, frame_id, timestamp, elapsed, np.mean(total_time)))

        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    print('Saved metric prefix:', result_metric)
    print('Saved video:', result_video[:-4] + '_' + args.mode + result_video[-4:])


def make_parser():
    parser = argparse.ArgumentParser('DeepSORVF++ experiment with DeepSORVF-style loop')
    parser.add_argument('--data_path', type=str, default='./clip-01/')
    parser.add_argument('--result_path', type=str, default='./result_deepsorvfpp_mainstyle/')
    parser.add_argument('--mode', default='full', choices=['botsort', 'virtual', 'assoc', 'full'])
    parser.add_argument('--max_frames', type=int, default=-1)
    parser.add_argument('--log_interval', type=int, default=30)
    parser.add_argument('--show_size', type=int, default=500)

    parser.add_argument('-f', '--exp_file', default=None, type=str)
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

    parser.add_argument('--cmc-method', dest='cmc_method', default='none', type=str)
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--name', default='clip-01')

    parser.add_argument('--with-reid', dest='with_reid', action='store_true')
    parser.add_argument('--fast-reid-config', dest='fast_reid_config',
                        default='BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml')
    parser.add_argument('--fast-reid-weights', dest='fast_reid_weights',
                        default='BoT-SORT/pretrained/mot17_sbs_S50.pth')
    parser.add_argument('--proximity_thresh', type=float, default=0.5)
    parser.add_argument('--appearance_thresh', type=float, default=0.25)

    parser.add_argument('--ais-max-age', dest='ais_max_age', type=float, default=2.0)
    parser.add_argument('--ais-kappa', dest='ais_kappa', type=float, default=0.5)
    parser.add_argument('--ais-position-var', dest='ais_position_var', type=float, default=4.0)
    parser.add_argument('--ais-scale-var', dest='ais_scale_var', type=float, default=1000000.0)
    parser.add_argument('--ais-bind-distance', dest='ais_bind_distance', type=float, default=120.0)
    parser.add_argument('--ais-cost-weight', dest='ais_cost_weight', type=float, default=0.25)
    parser.add_argument('--ais-heading-weight', dest='ais_heading_weight', type=float, default=0.05)
    parser.add_argument('--ais-occlusion-min-score', dest='ais_occlusion_min_score', type=float, default=0.4)
    parser.add_argument('--ais-occlusion-max-frames', dest='ais_occlusion_max_frames', type=int, default=60)
    parser.add_argument('--ais-cmc-mode', dest='ais_cmc_mode',
                        choices=['none', 'same', 'inverse'], default='inverse')
    parser.add_argument('--fps', type=int, default=30)
    parser.set_defaults(oar_polygon=None)
    return parser


if __name__ == '__main__':
    run(make_parser().parse_args())
