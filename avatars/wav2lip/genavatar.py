from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, gc
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import pickle
from avatars.wav2lip import face_detection

device = torch.device('cpu')
print('Using {} for dataset generation (fast memory-streaming mode).'.format(device))

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

def video2imgs(vid_path, save_path, ext = '.png', cut_frame = 10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            # Upstream burned a "LiveTalking" watermark into every extracted
            # frame here.  It is permanent — it survives into the served video
            # because these PNGs are the render background — and the runtime
            # already has an optional overlay (--debug_label) for the same
            # purpose, so the source frames stay clean.
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break
    cap.release()

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def generate_avatar(video_path, avatar_id, save_path='./data/avatars', img_size=256, pads=[0, 10, 0, 0], nosmooth=False, face_det_batch_size=8, det_size=640, progress_callback=None):
    """生成 avatar 的核心逻辑（内存流式，人脸检测按等比 letterbox 缩放）。

    ``det_size`` 是送入 S3FD 的方形画布边长；设为 0 表示按原始分辨率检测，
    精度最高但最慢。无论哪种方式，缩放都保持长宽比——把非方形帧直接压成
    正方形会让检测框在映射回原图后按同样比例失真。
    """
    avatar_path = os.path.join(save_path, avatar_id)
    full_imgs_path = os.path.join(avatar_path, "full_imgs")
    face_imgs_path = os.path.join(avatar_path, "face_imgs")
    coords_path = os.path.join(avatar_path, "coords.pkl")

    osmakedirs([avatar_path, full_imgs_path, face_imgs_path])

    if progress_callback: progress_callback(5)

    print(f"正在处理视频: {video_path}")
    input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
    if len(input_img_list) == 0:
        video2imgs(video_path, full_imgs_path, ext='png')
        input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

    total_frames = len(input_img_list)

    if progress_callback: progress_callback(20)

    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False,
        device=device
    )

    batch_size = face_det_batch_size
    predictions = []

    # 获得原图尺寸
    first_frame = cv2.imread(input_img_list[0])
    orig_h, orig_w = first_frame.shape[:2]
    del first_frame

    # 等比 letterbox：一个 scale 同时作用于两个轴，检测框换算回原图时不失真。
    # 早期实现用 cv2.resize(img, (640, 640)) 把 704x896 压成正方形，纵向比横向
    # 多压 27%，映射回来的每个框都因此偏高，Wav2Lip 拿到的人脸裁剪远非方形。
    if det_size and det_size > 0:
        scale = min(det_size / float(orig_w), det_size / float(orig_h))
        det_w, det_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        pad_x, pad_y = (det_size - det_w) // 2, (det_size - det_h) // 2
        print(f"共发现 {total_frames} 帧图片，按 {det_size}px letterbox 检测人脸"
              f"（{det_w}x{det_h}，缩放 {scale:.3f}）...")
    else:
        scale, det_w, det_h, pad_x, pad_y = 1.0, orig_w, orig_h, 0, 0
        print(f"共发现 {total_frames} 帧图片，按原始分辨率 {orig_w}x{orig_h} 检测人脸...")

    for i in tqdm(range(0, total_frames, batch_size), desc="流式人脸检测"):
        batch_paths = input_img_list[i : i + batch_size]
        batch_resized = []
        for p in batch_paths:
            img = cv2.imread(p)
            if scale == 1.0:
                batch_resized.append(img)
                continue
            canvas = np.zeros((det_size, det_size, 3), dtype=img.dtype)
            canvas[pad_y:pad_y + det_h, pad_x:pad_x + det_w] = cv2.resize(
                img, (det_w, det_h)
            )
            batch_resized.append(canvas)
            del img

        preds = detector.get_detections_for_batch(np.asarray(batch_resized))

        # 换算坐标回原图尺寸
        for pred in preds:
            if pred is None:
                predictions.append(None)
            else:
                x1, y1, x2, y2 = pred
                orig_x1 = max(0, min(orig_w, int(round((x1 - pad_x) / scale))))
                orig_y1 = max(0, min(orig_h, int(round((y1 - pad_y) / scale))))
                orig_x2 = max(0, min(orig_w, int(round((x2 - pad_x) / scale))))
                orig_y2 = max(0, min(orig_h, int(round((y2 - pad_y) / scale))))
                predictions.append((orig_x1, orig_y1, orig_x2, orig_y2))

        del batch_resized
        gc.collect()

        if progress_callback:
            progress = 20 + int((i + len(batch_paths)) / total_frames * 60)
            progress_callback(min(progress, 80))

    results = []
    pady1, pady2, padx1, padx2 = pads

    for rect in predictions:
        if rect is None:
            rect = [0, 0, orig_w, orig_h]

        y1 = max(0, rect[1] - pady1)
        y2 = min(orig_h, rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(orig_w, rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)

    if progress_callback: progress_callback(85)

    coord_list = []
    print("正在保存人脸图片和坐标（单帧流式裁剪）...")
    for idx, (rect, img_path) in enumerate(zip(boxes, input_img_list)):
        frame = cv2.imread(img_path)
        y1, y2, x1, x2 = int(rect[1]), int(rect[3]), int(rect[0]), int(rect[2])
        face_frame = frame[y1:y2, x1:x2]

        if face_frame.size > 0:
            resized_crop_frame = cv2.resize(face_frame, (img_size, img_size))
        else:
            resized_crop_frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", resized_crop_frame)
        coord_list.append((y1, y2, x1, x2))
        del frame

    print(f"写入坐标文件: {coords_path}")
    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)

    del detector
    gc.collect()

    if progress_callback: progress_callback(100)
    print("✅ Avatar 数据处理完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--avatar_id', default='wav2lip256_avatar1', type=str)
    parser.add_argument('--save_path', default='data/avatars', type=str)
    parser.add_argument('--video_path', default='', type=str)
    parser.add_argument('--nosmooth', default=False, action='store_true')
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0])
    parser.add_argument('--face_det_batch_size', type=int, default=8)
    parser.add_argument('--det_size', default=640, type=int,
                        help='人脸检测方形画布边长；0 表示按原始分辨率检测')
    args = parser.parse_args()

    generate_avatar(
        video_path=args.video_path,
        avatar_id=args.avatar_id,
        save_path=args.save_path,
        img_size=args.img_size,
        pads=args.pads,
        nosmooth=args.nosmooth,
        face_det_batch_size=args.face_det_batch_size,
        det_size=args.det_size,
    )
