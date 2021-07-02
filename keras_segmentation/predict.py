import glob
import random
import json
import os
import six

import cv2
import numpy as np
from tqdm import tqdm
from time import time

import statistics
import math
import warnings
from scipy.signal import find_peaks

from .train import find_latest_checkpoint
from .data_utils.data_loader import get_image_array, get_segmentation_array,\
    DATA_LOADER_SEED, class_colors, get_pairs_from_paths
from .models.config import IMAGE_ORDERING

random.seed(DATA_LOADER_SEED)

def model_from_checkpoint_path(checkpoints_path):

    from .models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    status = model.load_weights(latest_weights)

    if status is not None:
        status.expect_partial()

    return model


def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img


def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend


def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h), interpolation=cv2.INTER_NEAREST)

    fused_img = (0.7*inp_img + 0.3*seg_img).astype('uint8')
    return fused_img


def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img


def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None):

    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        original_h = inp_img.shape[0]
        original_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height), interpolation=cv2.INTER_NEAREST)
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)

        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img


def predict(model=None, inp=None, out_fname=None,
            checkpoints_path=None, overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None,
            read_image_type=1):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp, read_image_type)

    assert (len(inp.shape) == 3 or len(inp.shape) == 1 or len(inp.shape) == 4), "Image should be h,w,3 "

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height,
                        ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)

    seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=colors, overlay_img=overlay_img,
                                     show_legends=show_legends,
                                     class_names=class_names,
                                     prediction_width=prediction_width,
                                     prediction_height=prediction_height)

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return pr


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     class_names=None, show_legends=False, colors=class_colors,
                     prediction_width=None, prediction_height=None, read_image_type=1):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
            glob.glob(os.path.join(inp_dir, "*.jpeg"))
        inps = sorted(inps)

    assert type(inps) is list

    all_prs = []

    if not out_dir is None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.splitext(os.path.basename(inp))[0] + ".png")
            else:
                out_fname = os.path.join(out_dir, str(i) + ".png")

        pr = predict(model, inp, out_fname,
                     overlay_img=overlay_img, class_names=class_names,
                     show_legends=show_legends, colors=colors,
                     prediction_width=prediction_width,
                     prediction_height=prediction_height, read_image_type=read_image_type)

        all_prs.append(pr)

    return all_prs


def set_video(inp, video_name):
    cap = cv2.VideoCapture(inp)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (video_width, video_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #fourcc = cv2.VideoWriter_fourcc('a','v','c','1')
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    return cap, video, fps


def predict_video(model=None, inp=None, output=None,
                  checkpoints_path=None, display=False, overlay_img=True,
                  class_names=None, show_legends=False, colors=class_colors,
                  prediction_width=None, prediction_height=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)
    n_classes = model.n_classes

    cap, video, fps = set_video(inp, output)
    while(cap.isOpened()):
        prev_time = time()
        ret, frame = cap.read()
        if frame is not None:
            gray_1C = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_3C = cv2.cvtColor(gray_1C, cv2.COLOR_GRAY2BGR)
            pr = predict(model=model, inp=gray_3C)
            fused_img = visualize_segmentation(
                pr, frame, n_classes=n_classes,
                colors=colors,
                overlay_img=overlay_img,
                show_legends=show_legends,
                class_names=class_names,
                prediction_width=prediction_width,
                prediction_height=prediction_height
                )
        else:
            break
        #print("FPS: {}".format(1/(time() - prev_time)))
        if output is not None:
            fused_img = np.uint8(fused_img)
            video.write(fused_img)
        if display:
            cv2.imshow('Frame masked', fused_img)
            if cv2.waitKey(fps) & 0xFF == ord('q'):
                break
    cap.release()
    if output is not None:
        video.release()
    cv2.destroyAllWindows()


def evaluate(model=None, inp_images=None, annotations=None,
             inp_images_dir=None, annotations_dir=None, checkpoints_path=None, read_image_type=1):

    if model is None:
        assert (checkpoints_path is not None),\
                "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)

    if inp_images is None:
        assert (inp_images_dir is not None),\
                "Please provide inp_images or inp_images_dir"
        assert (annotations_dir is not None),\
            "Please provide inp_images or inp_images_dir"

        paths = get_pairs_from_paths(inp_images_dir, annotations_dir)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    assert type(inp_images) is list
    assert type(annotations) is list

    tp = np.zeros(model.n_classes)
    fp = np.zeros(model.n_classes)
    fn = np.zeros(model.n_classes)
    n_pixels = np.zeros(model.n_classes)

    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp, read_image_type=read_image_type)
        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height,
                                    no_reshape=True, read_image_type=read_image_type)
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()

        for cl_i in range(model.n_classes):

            tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
            fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
            fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
            n_pixels[cl_i] += np.sum(gt == cl_i)

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)

    return {
        "frequency_weighted_IU": frequency_weighted_IU,
        "mean_IU": mean_IU,
        "class_wise_IU": cl_wise_score
    }

def _check_overlap(line1, line2):
    combination = np.array([line1,
                            line2,
                            [line1[0], line1[1], line2[0], line2[1]],
                            [line1[0], line1[1], line2[2], line2[3]],
                            [line1[2], line1[3], line2[0], line2[1]],
                            [line1[2], line1[3], line2[2], line2[3]]])
    distance = np.sqrt((combination[:,0] - combination[:,2])**2 
    + (combination[:,1] - combination[:,3])**2)
    max = np.amax(distance)
    overlap = distance[0] + distance[1] - max
    endpoint = combination[np.argmax(distance)]
    return (overlap >= 0), endpoint #replace 0 with the value of distance between 2 collinear lines

def _merge_lines(line_list):
    #convert (x1, y1, x2, y2) form to (r, alpha) form
    if line_list is not None:
        line_list = np.squeeze(line_list, axis=1)
        line_list = np.array(line_list)
        A = line_list[:,1] - line_list[:,3]
        B = line_list[:,2] - line_list[:,0]
        C = line_list[:,0]*line_list[:,3] - line_list[:,2]*line_list[:,1]
        r = np.divide(np.abs(C), np.sqrt(A*A+B*B))
        alpha = (np.arctan2(-B,-A) + math.pi) % (2*math.pi) - math.pi
        r_alpha = np.column_stack((r, alpha))

        #prepare some variables to keep track of lines looping
        r_bin_size = 10 #maximum distance to treat 2 lines as one
        alpha_bin_size = 0.15 #maximum angle (radian) to treat 2 lines as one
        merged = np.zeros(len(r_alpha), dtype=np.uint8)
        line_group = np.empty((0,4), dtype=np.int32)
        group_count = 0

        for line_index in range(len(r_alpha)): 
            if merged[line_index] == 0: #if line hasn't been merged yet
                merged[line_index] = 1
                line_group = np.append(line_group, [line_list[line_index]], axis=0)
                for line_index2 in range(line_index+1,len(r_alpha)):
                    if merged[line_index2] == 0:
                        #calculate the differences between 2 lines by r and alpha
                        dr = abs(r_alpha[line_index,0] - r_alpha[line_index2,0])
                        dalpha = abs(r_alpha[line_index,1] - r_alpha[line_index2,1])
                        if (dr<r_bin_size) and (dalpha<alpha_bin_size): # if close, check overlap
                            overlap, endpoints = _check_overlap(line_group[group_count], line_list[line_index2])
                            if overlap:
                                line_group[group_count] = endpoints
                                merged[line_index2] = 1
                group_count += 1
        line_group = np.expand_dims(line_group, axis=1)
        return line_group
    else:
        return None 



def _get_egolanes_points(mask):
    h, w, _ = mask.shape
    image_l = (mask % 2)*100
    image_r = (mask // 2)*100
    uint8_img_l = np.uint8(image_l[:,:,0])
    uint8_img_r = np.uint8(image_r[:,:,0])

    lines_l = _merge_lines(cv2.HoughLinesP(uint8_img_l, 1, np.pi/30, 50, minLineLength=20, maxLineGap=10))
    lines_r = _merge_lines(cv2.HoughLinesP(uint8_img_r, 1, np.pi/30, 50, minLineLength=20, maxLineGap=10))
                                          
    intersection_pts = 0
    list_vx = []
    list_vy = []
    list_x1 = []
    list_x2 = []
    list_lw = [] # lane width
    mode_vx, mode_vy = None,None
    x1_intercept_mode, x2_intercept_mode = None, None

    # if both the lines are predicted
    if lines_l is not None and lines_r is not None:
        for line_l in lines_l:
            for line_r in lines_r:
                x1_l, y1_l, x2_l, y2_l = line_l[0]
                x1_r, y1_r, x2_r, y2_r = line_r[0]
                l_slope, l_intercept = np.polyfit((x1_l, x2_l), (y1_l, y2_l), 1)
                r_slope, r_intercept = np.polyfit((x1_r, x2_r), (y1_r, y2_r), 1)
        vx = int((r_intercept - l_intercept) / (l_slope - r_slope))
        vy = int(l_slope*((r_intercept - l_intercept)/ (l_slope - r_slope)) + l_intercept)
        x1_intercept = int((h-1 - l_intercept)/l_slope)
        x2_intercept = int((h-1 - r_intercept)/r_slope)
        # append valid intercepts to the list
        if x1_intercept > -3*w and x2_intercept > -3*w and x1_intercept < 3*w and x2_intercept < 3*w:
            list_vx.append(vx)
            list_vy.append(vy)
            list_x1.append(x1_intercept)
            list_x2.append(x2_intercept)
            list_lw.append(abs(x2_intercept - x1_intercept))
            intersection_pts += 1
        # calculate vx based on mode
        f_vx, b_vx = np.histogram(list_vx)
        mode_vx = int(b_vx[np.argmax(f_vx)])
        # calculate vy based on mode
        f_vy, b_vy = np.histogram(list_vy)
        mode_vy = int(b_vy[np.argmax(f_vy)])
        # find x1 intercept based on mode
        f_x1, b_x1 = np.histogram(list_x1, bins=np.arange(-3*w,3*w,w/20))
        x1_intercept_mode = int(b_x1[np.argmax(f_x1)])
        x1_intercept_multiple_modes = [b_x1[idx] for idx in find_peaks(f_x1, height=20)[0]]
        # find x2 intercept based on mode
        f_x2, b_x2 = np.histogram(list_x2, bins=np.arange(-3*w,3*w,w/20))
        x2_intercept_mode = int(b_x2[np.argmax(f_x2)])
        x2_intercept_multiple_modes = [b_x2[idx] for idx in find_peaks(f_x2, height=20)[0]]
    else:
        try:
            if lines_l is not None:
                for line_l in lines_l:
                    x1_l, y1_l, x2_l, y2_l = line_l[0]
                    l_slope, l_intercept = np.polyfit((x1_l, x2_l), (y1_l, y2_l), 1)
                    x1_intercept = int((h-1 - l_intercept)/l_slope)
                    vx =  int((0 - l_intercept)/l_slope)
                    if x1_intercept > -3*w and x1_intercept < 3*w:
                        list_x1.append(x1_intercept)
                        list_vx.append(vx)
            f_x1, b_x1 = np.histogram(list_x1, bins=np.arange(-3*w,3*w,w/20))
            x1_intercept_mode = int(b_x1[np.argmax(f_x1)])
            # calculate vx based on mode
            f_vx, b_vx = np.histogram(list_vx)
            mode_vx = int(b_vx[np.argmax(f_vx)])
            x1_intercept_multiple_modes = [b_x1[idx] for idx in find_peaks(f_x1, height=20)[0]]
        except Exception as e:
            print("Error finding x1 intercepts when x2 is None: ", e)
        try:
            if lines_r is not None:
                for line_r in lines_r:
                    x1_r, y1_r, x2_r, y2_r = line_r[0]
                    r_slope, r_intercept = np.polyfit((x1_r, x2_r), (y1_r, y2_r), 1)
                    x2_intercept = int((h-1 - r_intercept)/r_slope)
                    vx =  int((0 - r_intercept)/r_slope) 
                if x2_intercept > -3*w and x2_intercept < 3*w:
                        list_x2.append(x2_intercept)
            f_x2, b_x2 = np.histogram(list_x2, bins=np.arange(-3*w,3*w,w/20))
            x2_intercept_mode = int(b_x2[np.argmax(f_x2)])
            # calculate vx based on mode
            f_vx, b_vx = np.histogram(list_vx)
            mode_vx = int(b_vx[np.argmax(f_vx)])
            x1_intercept_multiple_modes = [b_x1[idx] for idx in find_peaks(f_x1, height=20)[0]]
            x2_intercept_multiple_modes = [b_x2[idx] for idx in find_peaks(f_x2, height=20)[0]]
        except Exception as e:
            print("Error finding x2 intercepts when x1 is None: ", e)
    
    results = {"x1_intercept_mode": x1_intercept_mode, 
    "x2_intercept_mode": x2_intercept_mode,
    "x1_intercept_multiple_modes": x1_intercept_multiple_modes,
    "x2_intercept_multiple_modes": x2_intercept_multiple_modes,
    "vx_mode": mode_vx,
    "vy_mode": mode_vy}
    return results


def evaluate_egolanes(model=None, inp_images=None, annotations=None,
                      inp_images_dir=None, annotations_dir=None, out_dir=None, checkpoints_path=None, read_image_type=1):

    if model is None:
        assert (checkpoints_path is not None),\
                "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)

    if inp_images is None:
        assert (inp_images_dir is not None),\
                "Please provide inp_images or inp_images_dir"
        assert (annotations_dir is not None),\
            "Please provide inp_images or inp_images_dir"

        paths = get_pairs_from_paths(inp_images_dir, annotations_dir)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    assert type(inp_images) is list
    assert type(annotations) is list

    errors = []
    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp, read_image_type=read_image_type)
        
        gt = get_segmentation_array(ann, model.n_classes, model.output_width, model.output_height, no_reshape=True, read_image_type=read_image_type) #one hot encoded array
        gt = gt.argmax(-1)
        
        inp_img = cv2.imread(inp)
        inp_h, inp_w, _ = inp_img.shape

        seg_out_pr = visualize_segmentation(pr, inp_img, n_classes=model.n_classes, colors=[(0, 0, 0), (1, 1, 1), (2, 2, 2)])
        seg_out_gt = visualize_segmentation(gt, inp_img, n_classes=model.n_classes, colors=[(0, 0, 0), (1, 1, 1), (2, 2, 2)])
        # call a function to get x1, x2 and vx and vy for both and gt and pr, and find the error
        egolanes_pts_gt = _get_egolanes_points(seg_out_gt)
        egolanes_pts_pr = _get_egolanes_points(seg_out_pr)

        error = 0
        if egolanes_pts_gt["x1_intercept_mode"] is not None and egolanes_pts_pr["x1_intercept_mode"] is not None:
            error += (egolanes_pts_gt["x1_intercept_mode"] - egolanes_pts_pr["x1_intercept_mode"])**2
        if egolanes_pts_gt["x2_intercept_mode"] is not None and egolanes_pts_pr["x2_intercept_mode"] is not None:
            error += (egolanes_pts_gt["x2_intercept_mode"] - egolanes_pts_pr["x2_intercept_mode"])**2
        if egolanes_pts_gt["vx_mode"] is not None and egolanes_pts_pr["vx_mode"] is not None:
            error += (egolanes_pts_gt["vx_mode"] - egolanes_pts_pr["vx_mode"])**2
        if egolanes_pts_gt["vy_mode"] is not None and egolanes_pts_pr["vy_mode"] is not None:
            error += (egolanes_pts_gt["vy_mode"] - egolanes_pts_pr["vy_mode"])**2
        error = math.sqrt(error)
        errors.append(error)
        #print("\nEgolanes GT", egolanes_pts_gt)
        #print("Egolanes PR", egolanes_pts_pr)
        #print("-"*50)
        
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"e-{int(error):05}_{os.path.basename(inp)}")
            
	    #segmentation output overlayed on input image
            fused_img = visualize_segmentation(pr, inp_img, n_classes=model.n_classes, overlay_img=True, 
                                               colors=[(0, 0, 0), (0, 0, 255), (0, 0, 125)],
                                               show_legends=False, class_names=["B", "L", "R"])
            # code to annotate image with the ground truth
            if egolanes_pts_gt["vx_mode"] is not None and egolanes_pts_gt["vy_mode"] is not None:
                inp_img = cv2.circle(inp_img, (egolanes_pts_gt["vx_mode"], egolanes_pts_gt["vy_mode"]), 2, (0, 255, 0), 2)
                cv2.line(inp_img, (egolanes_pts_gt["vx_mode"], 0), (egolanes_pts_gt["vx_mode"], inp_h-1), (0, 255, 0)) 
                cv2.line(inp_img, (0, egolanes_pts_gt["vy_mode"]), (inp_w-1, egolanes_pts_gt["vy_mode"]), (0, 255, 0))
                cv2.line(inp_img, (egolanes_pts_gt["x1_intercept_mode"], inp_h-1), (egolanes_pts_gt["vx_mode"], egolanes_pts_gt["vy_mode"]), (0, 255, 0), 2)
                cv2.line(inp_img, (egolanes_pts_gt["x2_intercept_mode"], inp_h-1), (egolanes_pts_gt["vx_mode"], egolanes_pts_gt["vy_mode"]), (0, 255, 0), 2)
            elif egolanes_pts_gt["x1_intercept_mode"] is not None:
                cv2.line(inp_img, (egolanes_pts_gt["x1_intercept_mode"], inp_h-1), (egolanes_pts_gt["vx_mode"], 0), (0, 255, 0), 2)
            elif egolanes_pts_gt["x2_intercept_mode"] is not None: 
                cv2.line(inp_img, (egolanes_pts_gt["x2_intercept_mode"], inp_h-1), (egolanes_pts_gt["vx_mode"], 0), (0, 255, 0), 2)
            # code to annotate image with the predictions
            if egolanes_pts_pr["vx_mode"] is not None and egolanes_pts_pr["vy_mode"] is not None:
                inp_img = cv2.circle(inp_img, (egolanes_pts_pr["vx_mode"], egolanes_pts_pr["vy_mode"]), 2, (0, 0, 255), 2)
                cv2.line(inp_img, (egolanes_pts_pr["vx_mode"], 0), (egolanes_pts_pr["vx_mode"], inp_h-1), (0, 0, 255)) 
                cv2.line(inp_img, (0, egolanes_pts_pr["vy_mode"]), (inp_w-1, egolanes_pts_pr["vy_mode"]), (0, 0, 255))
                cv2.line(inp_img, (egolanes_pts_pr["x1_intercept_mode"], inp_h-1), (egolanes_pts_pr["vx_mode"], egolanes_pts_pr["vy_mode"]), (0, 0, 255), 2)
                cv2.line(inp_img, (egolanes_pts_pr["x2_intercept_mode"], inp_h-1), (egolanes_pts_pr["vx_mode"], egolanes_pts_pr["vy_mode"]), (0, 0, 255), 2)
            elif egolanes_pts_pr["x1_intercept_mode"] is not None:
                cv2.line(inp_img, (egolanes_pts_pr["x1_intercept_mode"], inp_h-1), (egolanes_pts_pr["vx_mode"], 0), (0, 0, 255), 2)
            elif egolanes_pts_pr["x2_intercept_mode"] is not None: 
                cv2.line(inp_img, (egolanes_pts_pr["x2_intercept_mode"], inp_h-1), (egolanes_pts_pr["vx_mode"], 0), (0, 0, 255), 2)
            out_img = np.concatenate([fused_img, inp_img], axis=1)
            cv2.imwrite(out_path, out_img) 
    return np.mean(errors)
