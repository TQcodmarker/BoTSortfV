import numpy as np
import pandas as pd
from warnings import simplefilter

try:
    from IPython import embed
except ImportError:
    def embed():
        return None


simplefilter(action='ignore', category=FutureWarning)


def _df_append(df, row, ignore_index=False):
    return pd.concat([df, pd.DataFrame([row])], ignore_index=ignore_index)


def _clean_vis_dataframe(df, include_speed=False):
    columns = ['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y']
    if include_speed:
        columns.append('speed')
    columns.append('timestamp')

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=columns)

    df = df.reindex(columns=columns).copy()
    numeric_columns = [col for col in columns if col != 'speed']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_columns)
    if len(df) == 0:
        return pd.DataFrame(columns=columns)

    finite_mask = np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all(axis=1)
    df = df[finite_mask]
    df = df[(df['x2'] > df['x1']) & (df['y2'] > df['y1'])]
    if len(df) == 0:
        return pd.DataFrame(columns=columns)

    for col in numeric_columns:
        df[col] = df[col].astype(int)
    if include_speed:
        df['speed'] = df['speed'].fillna('[0, 0]')
    return df.reset_index(drop=True)


def box_whether_in_area(bounding_box, Area):
    x_center = (bounding_box[0] + bounding_box[2]) / 2
    y_center = (bounding_box[1] + bounding_box[3]) / 2
    Area = [1] + Area
    return whether_in_area((x_center, y_center), Area)


def speed_extract(last_traj, now_traj):
    last_x = int(last_traj.loc['x'])
    last_y = int(last_traj.loc['y'])
    cur_x = int(now_traj.loc['x'])
    cur_y = int(now_traj.loc['y'])
    x_speed = (cur_x - last_x) / (int(now_traj.loc['timestamp']) - int(last_traj.loc['timestamp']))
    y_speed = (cur_y - last_y) / (int(now_traj.loc['timestamp']) - int(last_traj.loc['timestamp']))
    return [x_speed, y_speed]


def whether_in_area(point, bbox):
    if point[0] <= bbox[3] and point[0] >= bbox[1] and point[1] <= bbox[4] and point[1] >= bbox[2]:
        return 1
    else:
        return 0


def overlap(box1, box2, val):
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:
        return 0
    else:
        max_x1 = max(minx1, minx2)
        min_x2 = min(maxx1, maxx2)
        max_y1 = max(miny1, miny2)
        min_y2 = min(maxy1, maxy2)
        Cross_area = (min_x2 - max_x1) * (min_y2 - max_y1)
        box1_area = (maxx1 - minx1) * (maxy1 - miny1)
        box2_area = (maxx2 - minx2) * (maxy2 - miny2)
        if Cross_area / box1_area > val or Cross_area / box2_area > val:
            return 1
        else:
            return 0


def whether_occlusion(bbox, cur_bbox_list, val):
    occlusion_bbox_list = []
    occlusion_id_list = []
    for i in range(len(cur_bbox_list)):
        flag = overlap(bbox[1:], cur_bbox_list[i][1:], val)
        if flag:
            if len(occlusion_id_list) == 0:
                occlusion_id_list.append(bbox[0])
                occlusion_bbox_list.append(bbox[1:])
            occlusion_bbox_list.append(cur_bbox_list[i][1:])
            occlusion_id_list.append(cur_bbox_list[i][0])
            break
    return occlusion_bbox_list, occlusion_id_list


def whether_in_OAR(point, OAR_list):
    flag = 0
    for oar in OAR_list:
        oar_id = [0, oar[0], oar[1], oar[2], oar[3]]
        if whether_in_area(point, oar_id):
            flag = whether_in_area(point, oar_id)
            break
    return flag


def OAR_extractor(his_traj_dataframe_list, val):
    OAR_list = []
    OAR_id_list = []
    if len(his_traj_dataframe_list) == 0:
        return OAR_list, OAR_id_list
    his_id_list = his_traj_dataframe_list[-1]['ID'].unique()
    his_bbox_list = []
    for i in range(len(his_id_list)):
        visual_traj = his_traj_dataframe_list[-1].iloc[i]
        his_bbox_list.append([visual_traj['ID'], visual_traj['x1'], visual_traj['y1'], visual_traj['x2'],
                              visual_traj['y2']])
    for i in range(len(his_bbox_list)):
        if i < len(his_bbox_list) - 1:
            occlusion_boxes, occlusion_ids = whether_occlusion(his_bbox_list[i], his_bbox_list[i + 1:], val)
            for index in range(len(occlusion_boxes)):
                if (occlusion_ids[index] not in OAR_id_list) and (occlusion_ids[index] in his_id_list):
                    OAR_list.append(occlusion_boxes[index])
                    OAR_id_list.append(occlusion_ids[index])
    return OAR_list, OAR_id_list


def motion_features_extraction(his_traj_dataframe_list, VIS_tra_cur):
    speed_list = []
    VIS_traj_cur_withfeature = VIS_tra_cur.copy()
    cur_id_list = VIS_tra_cur['ID'].unique()
    for i in range(len(cur_id_list)):
        speed_list.append('[0, 0]')
    VIS_traj_cur_withfeature['speed'] = speed_list
    for k in range(len(cur_id_list)):
        if len(his_traj_dataframe_list) == 0:
            continue
        id = cur_id_list[k]
        for i in his_traj_dataframe_list:
            his_id_list = list(i['ID'].unique())
            if id not in his_id_list:
                continue
            else:
                index = his_id_list.index(id)
                last_traj = i.iloc[index]
                VIS_traj_cur_withfeature.loc[k, 'speed'] = str(speed_extract(last_traj, VIS_traj_cur_withfeature.iloc[k]))
                break
    return VIS_traj_cur_withfeature


def id_whether_stable(id, last_5_trajs):
    for traj in last_5_trajs:
        if id in list(traj['ID'].unique()):
            continue
        else:
            return False
    return True


class VISPRO_BoTSORT(object):
    def __init__(self, predictor, tracker, anti, val, t):
        self.predictor = predictor
        self.tracker = tracker
        self.anti = anti
        self.last5_vis_tra_list = []
        self.Vis_tra_cur_3 = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'timestamp'])
        self.Vis_tra_cur = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'speed', 'timestamp'])
        self.Vis_tra = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'speed', 'timestamp'])
        self.VIS_tra_last = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'speed', 'timestamp'])
        self.OAR_list = []
        self.OAR_ids_list = []
        self.OAR_mmsi_list = []
        self.val = val
        self.t = t
        self.Anti_occlusion_traj = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'speed', 'timestamp'])

    def detection(self, image):
        outputs, img_info = self.predictor.inference(image)
        if outputs[0] is None:
            return []

        ratio = img_info.get('ratio')
        if ratio is None:
            ratio = min(
                self.predictor.test_size[0] / float(img_info['height']),
                self.predictor.test_size[1] / float(img_info['width']))

        detections = outputs[0].detach().cpu().numpy()
        detections[:, :4] /= ratio

        bboxes = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            obj_conf = det[4]
            cls_conf = det[5] if det.shape[0] > 5 else 1.0
            conf = float(obj_conf * cls_conf)
            bboxes.append((float(x1), float(y1), float(x2), float(y2), 'vessel', conf))
        return bboxes

    @staticmethod
    def _boxes_to_botsort_detections(bboxes, force_conf=None):
        detections = []
        for x1, y1, x2, y2, _, conf in bboxes:
            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)
            if x2 <= x1 or y2 <= y1:
                continue
            score = float(conf if force_conf is None else force_conf)
            detections.append([x1, y1, x2, y2, score, 1.0, 0.0])
        if len(detections) == 0:
            return np.empty((0, 7), dtype=float)
        return np.asarray(detections, dtype=float)

    def track(self, image, bboxes, bboxes_anti_occ, id_list, timestamp):
        detections_normal = self._boxes_to_botsort_detections(bboxes)
        detections_anti_occ = self._boxes_to_botsort_detections(bboxes_anti_occ, force_conf=1.0)
        if len(detections_normal) and len(detections_anti_occ):
            detections = np.concatenate([detections_normal, detections_anti_occ], axis=0)
        elif len(detections_normal):
            detections = detections_normal
        else:
            detections = detections_anti_occ

        config = self.tracker.ais_config
        old_max_age = config.max_age
        old_cost_weight = config.cost_weight
        old_heading_weight = config.heading_weight
        old_occlusion_min_score = config.occlusion_min_score
        try:
            config.max_age = -1.0
            config.cost_weight = 0.0
            config.heading_weight = 0.0
            config.occlusion_min_score = float('inf')
            online_targets = self.tracker.update(detections, image, ais_frame=[], timestamp=timestamp)
        finally:
            config.max_age = old_max_age
            config.cost_weight = old_cost_weight
            config.heading_weight = old_heading_weight
            config.occlusion_min_score = old_occlusion_min_score

        for target in list(online_targets):
            tlwh = np.asarray(target.tlwh, dtype=float)
            if tlwh.shape[0] < 4 or not np.all(np.isfinite(tlwh[:4])):
                continue
            x1 = tlwh[0]
            y1 = tlwh[1]
            x2 = tlwh[0] + tlwh[2]
            y2 = tlwh[1] + tlwh[3]
            track_id = int(target.track_id)
            if track_id in id_list and id_list.index(track_id) < len(bboxes_anti_occ):
                x1, y1, x2, y2, _, _ = bboxes_anti_occ[id_list.index(track_id)]
            self.Vis_tra_cur_3 = _df_append(self.Vis_tra_cur_3, {
                'ID': track_id,
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
                'x': int((x1 + x2) / 2),
                'y': int((y1 + y2) / 2),
                'timestamp': timestamp // 1000
            }, ignore_index=True)

    def update_tra(self, Vis_tra, timestamp):
        self.Vis_tra_cur = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'timestamp'])
        id_list = self.Vis_tra_cur_3['ID'].unique()
        for k in range(len(id_list)):
            id_current = self.Vis_tra_cur_3[self.Vis_tra_cur_3['ID'] == id_list[k]].reset_index(drop=True)
            df = id_current.mean(numeric_only=True).astype(int)
            df['timestamp'] = timestamp // 1000
            self.Vis_tra_cur = _df_append(self.Vis_tra_cur, df, ignore_index=True)
        self.Vis_tra_cur_3 = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'timestamp'])
        self.Vis_tra_cur = _clean_vis_dataframe(self.Vis_tra_cur, include_speed=False)

        Vis_tra_cur_withfeature = motion_features_extraction(self.last5_vis_tra_list, VIS_tra_cur=self.Vis_tra_cur)
        Vis_tra_cur_withfeature = _clean_vis_dataframe(Vis_tra_cur_withfeature, include_speed=True)
        self.Vis_tra_cur = Vis_tra_cur_withfeature
        self.Vis_tra = pd.concat([self.Vis_tra, Vis_tra_cur_withfeature], ignore_index=True)
        self.Vis_tra = _clean_vis_dataframe(self.Vis_tra, include_speed=True)
        if len(self.last5_vis_tra_list) > 4:
            self.last5_vis_tra_list.pop(0)
        self.last5_vis_tra_list.append(Vis_tra_cur_withfeature)
        time_limited = 2
        self.Vis_tra = self.Vis_tra.drop(self.Vis_tra[self.Vis_tra['timestamp'] <
                                                      (timestamp // 1000 - time_limited * 60)].index)
        return Vis_tra_cur_withfeature

    def traj_prediction_via_visual(self, last_traj, timestamp, speed):
        Vis_tra_prediction = last_traj.copy()
        x_move = int(timestamp - last_traj.loc['timestamp']) * float(speed[0])
        y_move = int(timestamp - last_traj.loc['timestamp']) * float(speed[1])
        Vis_tra_prediction.loc['x'] = Vis_tra_prediction.loc['x'] + x_move
        Vis_tra_prediction.loc['x1'] = Vis_tra_prediction.loc['x1'] + x_move
        Vis_tra_prediction.loc['x2'] = Vis_tra_prediction.loc['x2'] + x_move
        Vis_tra_prediction.loc['y'] = Vis_tra_prediction.loc['y'] + y_move
        Vis_tra_prediction.loc['y1'] = Vis_tra_prediction.loc['y1'] + y_move
        Vis_tra_prediction.loc['y2'] = Vis_tra_prediction.loc['y2'] + y_move
        Vis_tra_prediction.loc['timestamp'] = timestamp

        return Vis_tra_prediction

    def anti_occ(self, last5_vis_tra_list, bboxes, AIS_vis, bind_inf, timestamp):
        bboxes_anti_occ = []
        if len(self.OAR_list):
            pop_index_list = []
            for index in range(len(bboxes)):
                for OAR in self.OAR_list:
                    if box_whether_in_area(bboxes[index][:4], OAR):
                        pop_index_list.append(index)
                        break
            for pop_index in range(len(pop_index_list)):
                bboxes.pop(pop_index_list[pop_index] - pop_index)

            bind_id_list = list(bind_inf['ID'].unique())
            self.OAR_mmsi_list = []
            OAR_ids_list_copy = self.OAR_ids_list.copy()
            for k in range(len(OAR_ids_list_copy)):
                if OAR_ids_list_copy[k] in bind_id_list:
                    mmsi = bind_inf.iloc[bind_id_list.index(OAR_ids_list_copy[k])].loc['mmsi']
                    self.OAR_mmsi_list.append([OAR_ids_list_copy[k], int(mmsi)])
                else:
                    self.OAR_mmsi_list.append([OAR_ids_list_copy[k], 0])

            ais_vis_mmsi_list = list(AIS_vis['mmsi'])
            pop_index_list = []
            for k in range(len(self.OAR_mmsi_list)):
                final_find_flg = 0
                second_final_find_flg = 0
                final_pos = []
                second_final_pos = []
                if not self.OAR_mmsi_list[k][1] == 0 and self.OAR_mmsi_list[k][1] in ais_vis_mmsi_list:
                    for i in range(len(ais_vis_mmsi_list)):
                        if int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['mmsi']) == self.OAR_mmsi_list[k][1] and \
                                int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['timestamp']) == timestamp - 1:
                            final_find_flg = 1
                            final_pos = [AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['x'],
                                         AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['y']]
                            continue
                        elif int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['mmsi']) == self.OAR_mmsi_list[k][
                            1] and int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['timestamp']) == timestamp - 2:
                            second_final_find_flg = 1
                            second_final_pos = [AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['x'],
                                                AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['y']]
                            continue
                        if final_find_flg and second_final_find_flg:
                            x_motion = final_pos[0] - second_final_pos[0]
                            y_motion = final_pos[1] - second_final_pos[1]
                            bboxes_anti_occ.append(
                                (self.Anti_occlusion_traj.iloc[k].loc['x1'] + x_motion,
                                 self.Anti_occlusion_traj.iloc[k].loc['y1'] + y_motion,
                                 self.Anti_occlusion_traj.iloc[k].loc['x2'] + x_motion,
                                 self.Anti_occlusion_traj.iloc[k].loc['y2'] + y_motion,
                                 'vessel', 1))
                            break
                else:
                    if not id_whether_stable(self.OAR_mmsi_list[k][0], last5_vis_tra_list):
                        pop_index_list.append(k)
                        continue
                    index = list(last5_vis_tra_list[0]['ID'].unique()).index(self.OAR_mmsi_list[k][0])
                    speed_str = last5_vis_tra_list[0].iloc[index].loc['speed']
                    speed = [float(speed_str[1:-1].split(',')[0]), float(speed_str[1:-1].split(',')[1])]
                    trajs = last5_vis_tra_list[0]
                    id_list = list(trajs['ID'].unique())
                    last_traj = trajs.iloc[id_list.index(self.OAR_mmsi_list[k][0])]
                    Vis_traj_now = self.traj_prediction_via_visual(last_traj, timestamp, speed)
                    bboxes_anti_occ.append(
                        (Vis_traj_now.loc['x1'],
                         Vis_traj_now.loc['y1'],
                         Vis_traj_now.loc['x2'],
                         Vis_traj_now.loc['y2'],
                         'vessel', 1))

            for i in range(len(pop_index_list)):
                self.OAR_mmsi_list.pop(pop_index_list[i] - i)
                self.OAR_ids_list.pop(pop_index_list[i] - i)
                self.OAR_list.pop(pop_index_list[i] - i)
            if not len(self.OAR_ids_list) == len(bboxes_anti_occ):
                embed()
        return bboxes_anti_occ

    def feedCap(self, image, timestamp, AIS_vis, bind_inf):
        if timestamp % 1000 < self.t:
            bboxes = self.detection(image)
            bboxes_anti_occ = self.anti_occ(self.last5_vis_tra_list, bboxes, AIS_vis, bind_inf, timestamp // 1000)

            self.track(image, bboxes, bboxes_anti_occ=bboxes_anti_occ,
                       id_list=self.OAR_ids_list, timestamp=timestamp // 1000)

            Vis_tra_cur = self.Vis_tra_cur
            if timestamp % 1000 < self.t:
                Vis_tra_cur = self.update_tra(self.Vis_tra, timestamp)
                if self.anti:
                    self.OAR_list, self.OAR_ids_list = OAR_extractor(self.last5_vis_tra_list, self.val)
                self.VIS_tra_last = Vis_tra_cur

                self.Anti_occlusion_traj = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'speed', 'timestamp'])
                id_list = list(self.VIS_tra_last['ID'].unique())
                for i in self.OAR_ids_list:
                    self.Anti_occlusion_traj = _df_append(self.Anti_occlusion_traj, self.VIS_tra_last.iloc[id_list.index(i)], ignore_index=True)
        self.Vis_tra = _clean_vis_dataframe(self.Vis_tra, include_speed=True)
        self.Vis_tra_cur = _clean_vis_dataframe(self.Vis_tra_cur, include_speed=True)
        return self.Vis_tra, self.Vis_tra_cur


VISPRO = VISPRO_BoTSORT
