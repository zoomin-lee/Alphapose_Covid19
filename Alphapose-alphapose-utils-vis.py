import math
import time

import cv2
import numpy as np
import torch

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]

    return color

def vis_frame_fast(frame, im_res, opt, format='coco'):
    print("1")
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii
    
    return rendered image
    '''

    kp_num = 17
    if len(im_res['result']) > 0:
    	kp_num = len(im_res['result'][0]['keypoints'])
    if kp_num == 17:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                          (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                          (77, 222, 255), (255, 156, 127),
                          (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
        elif format == 'mpii':
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
            p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        else:
            raise NotImplementedError
    elif kp_num == 136:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
            (26, 27),(27, 28),(28, 29),(29, 30),(30, 31),(31, 32),(32, 33),(33, 34),(34, 35),(35, 36),(36, 37),(37, 38),#Face
            (38, 39),(39, 40),(40, 41),(41, 42),(43, 44),(44, 45),(45, 46),(46, 47),(48, 49),(49, 50),(50, 51),(51, 52),#Face
            (53, 54),(54, 55),(55, 56),(57, 58),(58, 59),(59, 60),(60, 61),(62, 63),(63, 64),(64, 65),(65, 66),(66, 67),#Face
            (68, 69),(69, 70),(70, 71),(71, 72),(72, 73),(74, 75),(75, 76),(76, 77),(77, 78),(78, 79),(79, 80),(80, 81),#Face
            (81, 82),(82, 83),(83, 84),(84, 85),(85, 86),(86, 87),(87, 88),(88, 89),(89, 90),(90, 91),(91, 92),(92, 93),#Face
            (94,95),(95,96),(96,97),(97,98),(94,99),(99,100),(100,101),(101,102),(94,103),(103,104),(104,105),#LeftHand
            (105,106),(94,107),(107,108),(108,109),(109,110),(94,111),(111,112),(112,113),(113,114),#LeftHand
            (115,116),(116,117),(117,118),(118,119),(115,120),(120,121),(121,122),(122,123),(115,124),(124,125),#RightHand
            (125,126),(126,127),(115,128),(128,129),(129,130),(130,131),(115,132),(132,133),(133,134),(134,135)#RightHand
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    elif kp_num == 26:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
    
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    else:
        raise NotImplementedError
    # im_name = os.path.basename(im_res['imgname'])

    left_right_image = np.split(frame, 2, axis=1)
    left_rect = left_right_image[0]
    right_rect = left_right_image[1]

    max_disparity = 5 * 16
    min_disparity = 0
    num_disparities = max_disparity - min_disparity
    window_size = 5
    stereo = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                   numDisparities=num_disparities,
                                   blockSize=3,
                                   uniquenessRatio=10,
                                   speckleWindowSize=5,
                                   speckleRange=1,
                                   disp12MaxDiff=12,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2)
    stereo2 = cv2.ximgproc.createRightMatcher(stereo)

    lamb = 8000
    sig = 1.5
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(lamb)
    wls_filter.setSigmaColor(sig)

    disparity = stereo.compute(left_rect, right_rect)
    disparity2 = stereo2.compute(right_rect, left_rect)
    disparity2 = np.int16(disparity2)

    filtered = wls_filter.filter(disparity, left_right_image[0], None, disparity2)
    _, filtered = cv2.threshold(filtered, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    filtered = (filtered / 16)

    # filtered[filtered<=0] = 0
    # filteredImg = 255*((filtered - filtered.min())/(filtered.max()-filtered.min()))

    # filteredImg = filtered.astype(np.uint8)
    #
    # cv2.imshow("disparity map", filteredImg)

    ############## resolution = VGA ##############################
    fx = 264.61
    baseline = 0.12003
    depth = np.zeros(shape=filtered.shape).astype(float)
    depth[filtered > 0] = ((fx * baseline) / (filtered[filtered > 0]))

    depth[depth > 1.0] = depth[depth > 1.0] * 1.3
    depth[depth > 2.0] = depth[depth > 2.0] * 1.4

    img = left_rect.copy()

    height, width = img.shape[:2]
    depthlist = []
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        if kp_num == 17:
            kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
        if opt.pose_track or opt.tracking:
            color = get_color_fast(int(abs(human['idx'])))
        else:
            color = RED

        # Draw keypoints
        count = 0
        depths = []
        meanx = []
        meany = []
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.5:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)

            if n < len(p_color):
                if opt.tracking:
                    cv2.circle(img, (cor_x, cor_y), 3, color, -1)
                else:
                    cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
                    if cor_y < 376 and cor_x < 672:
                        count += 1
                        meanx.append(cor_x)
                        meany.append(cor_y)
                        depths += depth[int(cor_y), int(cor_x)]
            else:
                cv2.circle(img, (cor_x, cor_y), 1, (255, 255, 255), 2)

        # Draw bboxes
        if opt.showbox:
            for n in range(kp_scores.shape[0]):
                if kp_scores[n] <= 0.7:
                    continue
                else:
                    if count > 0:
                        depthlist.append(depths / count)
                        if depths / count < 1.0:
                            if 'box' in human.keys():
                                bbox = human['box']
                                bbox = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]]  # xmin,xmax,ymin,ymax

                            else:
                                from trackers.PoseFlow.poseflow_infer import get_box
                                keypoints = []
                                for n in range(kp_scores.shape[0]):
                                    keypoints.append(float(kp_preds[n, 0]))
                                    keypoints.append(float(kp_preds[n, 1]))
                                    keypoints.append(float(kp_scores[n]))
                                bbox = get_box(keypoints, height, width)
                            # color = get_color_fast(int(abs(human['idx'][0][0])))
                            cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), color, 2)
                            cv2.putText(img, "-- Warning -- depth : {:.2f} m --".format(
                                depths / count), (int(bbox[0]), int(bbox[2] - 12)), cv2.FONT_ITALIC,
                                        0.5, RED, 1, lineType=cv2.LINE_AA)
                            if opt.tracking:
                                cv2.putText(img, str(human['idx']), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT,
                                            1, BLACK, 2)
                        else:
                            if depths != "inf":
                                cv2.putText(img, "-- depth : {:.2f} m --".format(
                                    depths / count), (min(meanx), min(meany) - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, RED, 1, lineType=cv2.LINE_AA)


        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                if i < len(line_color):
                    if opt.tracking:
                        cv2.line(img, start_xy, end_xy, color, 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                    else:
                        cv2.line(img, start_xy, end_xy, line_color[i], 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                else:
                    cv2.line(img, start_xy, end_xy, (255,255,255), 1)  
        depth2person = min(depthlist)
    return img

def vis_frame(frame, im_res, opt, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    kp_num = 17
    if len(im_res['result']) > 0:
    	kp_num = len(im_res['result'][0]['keypoints'])

    if kp_num == 17:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]

            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                          (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                          (77, 222, 255), (255, 156, 127),
                          (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
        elif format == 'mpii':
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
            p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
            line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    # im_name = os.path.basename(im_res['imgname'])
    left_right_image = np.split(frame, 2, axis=1)
    left_rect = left_right_image[0]
    right_rect= left_right_image[1]

    max_disparity = 5*16
    min_disparity = -16
    num_disparities = max_disparity - min_disparity
    window_size = 5
    stereo = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                   numDisparities=num_disparities,
                                   blockSize=3,
                                   uniquenessRatio=10,
                                   speckleWindowSize=5,
                                   speckleRange=1,
                                   disp12MaxDiff=12,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2)

    stereo2 = cv2.ximgproc.createRightMatcher(stereo)

    lamb = 8000
    sig = 1.5
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(lamb)
    wls_filter.setSigmaColor(sig)

    disparity = stereo.compute(left_rect, right_rect)
    disparity2 = stereo2.compute(right_rect, left_rect)
    disparity2 = np.int16(disparity2)

    filtered = wls_filter.filter(disparity, left_right_image[0], None, disparity2)
    _, filtered = cv2.threshold(filtered, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    filtered = (filtered / 16)

    # filteredImg = filtered.astype(np.uint8)
    # cv2.imshow("disparity map", filteredImg)

############## resolution = VGA ##############################
    fx = 264.61
    baseline = 0.12003 # m unit
############## resolution = HD ##############################
    # fx = 529.22
    # baseline = 0.12003 # m unit

    depth = np.zeros(shape=filtered.shape).astype(float)
    depth[filtered > 0] = ((fx * baseline) / (filtered[filtered > 0]))

    depth[depth > 1.0] = depth[depth > 1.0] * 1.3
    depth[depth > 2.0] = depth[depth > 2.0] * 1.4
    img = left_rect.copy()

    height, width = img.shape[:2]
    z3d_list, x3d_list, y3d_list = [],[],[]

    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        if kp_num == 17:
            kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
        if opt.tracking:
            color = get_color_fast(int(abs(human['idx'])))
        else:
            color = RED

        # Draw keypoints
        count = 0
        depths = 0
        meanx = []
        meany = []
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.7:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            bg = img.copy()
            if n < len(p_color):
                if opt.tracking:
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 2, color, -1)
                else:
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
                    if cor_y < 376 and cor_x < 672:
                        count += 1
                        meanx.append(cor_x)
                        meany.append(cor_y)
                        depths += depth[int(cor_y), int(cor_x)]
            else:
                cv2.circle(bg, (int(cor_x), int(cor_y)), 1, (255, 255, 255), 2)

            # Now create a mask of logo and create its inverse mask also
            transparency = float(max(0, min(1, kp_scores[n])))
            img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
        if count != 0 :
            z3d_list.append(depths / count)
            ############## resolution = VGA ##############################
            x3d = ((sum(meanx) / count - 335.4425) * ((depths / count)) / fx)
            y3d = ((sum(meany) / count - 182.4035) * ((depths / count)) / fx)
            ############## resolution = HD ##############################
            # x3d = ((sum(meanx) / count - 626.88) * (depths / count)) / fx
            # y3d = ((sum(meany) / count - 363.6925) * (depths / count)) / fx

            # when showbox off
            if 'box' in human.keys():
                bbox = human['box']
                bbox = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]]  # xmin,xmax,ymin,ymax
                cv2.rectangle(img, (int(bbox[0]-20), int(bbox[2]-18)), (int(bbox[1]+70), int(bbox[2]-1)), (47,157,39), -1)
                cv2.putText(img, "( %.2f, %.2f, %.2f ) m "%( round(x3d,2), round(y3d,2), round(depths / count, 2)),
                            (int(bbox[0]-10), int(bbox[2] - 6)), cv2.FONT_HERSHEY_DUPLEX,0.4, (255,255,255), 1)

            x3d_list.append(x3d)
            y3d_list.append(y3d)
        else :
            z3d_list.append(500)
            x3d_list.append(500)
            y3d_list.append(500)

        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[ 1]))
                stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length/2), int(stickwidth)), int(angle), 0, 360, 1)
                if i < len(line_color):
                    if opt.tracking:
                        cv2.fillConvexPoly(bg, polygon, (255,255,255))
                    else:
                        cv2.fillConvexPoly(bg, polygon, (255,255,255))
                else:
                    cv2.line(bg, start_xy, end_xy, (255,255,255), 1)
                transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])-0.1)))
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)

    idx =[]
    for i in range(len(z3d_list)-1):
        for j in range(i+1, len(z3d_list)):
            if z3d_list[i] != 500:
                p2p = np.sqrt((z3d_list[i]-z3d_list[j])**2+(x3d_list[i]-x3d_list[j])**2+(y3d_list[i]-y3d_list[j])**2)
                if 0<=p2p<=1:
                    idx.append([i, j])

    if len(idx) >= 1 :
        for l in idx:
            human1 = im_res['result'][l[0]]
            human2 = im_res['result'][l[1]]

            bbox1 = human1['box']
            bbox1 = [bbox1[0], bbox1[0] + bbox1[2], bbox1[1], bbox1[1] + bbox1[3]]
            cv2.rectangle(img, (int(bbox1[0]-20), int(bbox1[2])), (int(bbox1[1]), int(bbox1[3])), color, 1)
            cv2.putText(img, "TOO CLOSE", (int(bbox1[0]), int(bbox1[2] - 20)), cv2.FONT_ITALIC,
                        0.5, RED, 1, lineType=cv2.LINE_AA)

            bbox2 = human2['box']
            bbox2 = [bbox2[0], bbox2[0] + bbox2[2], bbox2[1], bbox2[1] + bbox2[3]]
            cv2.rectangle(img, (int(bbox2[0]-20), int(bbox2[2])), (int(bbox[1]+70), int(bbox2[3])), color, 1)
            cv2.putText(img, "TOO CLOSE", (int(bbox2[0]), int(bbox2[2] - 20)), cv2.FONT_ITALIC,
                        0.5, RED, 1, lineType=cv2.LINE_AA)

    return img


def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval
