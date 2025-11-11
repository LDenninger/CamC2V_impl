"""
    Evaluation script for camera trajectory alignmet.
    This has to be run after the evaluate.py script that generated the videos.
"""
import os, sys
import argparse
from pathlib import Path
import subprocess
from termcolor import colored
import numpy as np
import torch
from torch import Tensor
import traceback 
import time
import yaml
from typing import Tuple, Literal
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
import imageio
import collections
import struct
import logging
import matplotlib.pyplot as plt
from glob import glob
import re
import math
from concurrent.futures import ThreadPoolExecutor

###########################################################
######################## Globals ##########################
###########################################################

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


###########################################################
################ Camera Pose Evaluation ###################
###########################################################

def camera_pose_evaluation(eval_paths,
                            output,
                            thread_id: int = 0,
                            use_colmap=True,
                            trials_per_video: int = 1,
                            trial_strategy: Literal["average", "best"] = "average",
                            ) -> Tuple[dict, list, list, list]:

    print(f"Thread {thread_id}: Starting camera pose evaluation for {len(eval_paths)} videos...")
    
    tmp_dir = Path(output) / f"tmp_thread_{thread_id}"
    os.makedirs(tmp_dir, exist_ok=True)

    save_dict = {}
    rot_err_list = []
    trans_err_list = []
    cam_mc_list = []
    
    for i, p in tqdm(enumerate(eval_paths), total=len(eval_paths), desc=f"Thread {thread_id}"):
        cam_data_file = Path(p) / "camera_params.npz"
        video_file = Path(p) / "raw" /"ground_truth.mp4"
        if not cam_data_file.exists():
            print(f"Thread {thread_id}: Could not find camera data for {Path(p).stem}. Skipping...")
            continue
        try:
            cam_data_gt = np.load(cam_data_file)
            #cam_data_gt = cam_data_gt[:, 1:] 
            gt_w2c = torch.from_numpy(cam_data_gt["RT"])
            gt_c2w = gt_w2c.inverse()
            #gt_w2c = cam_data_gt[:, 6:].reshape((-1, 3, 4))
            #gt_c2w = rt34_to_44(gt_w2c).inverse()
            gt_rel_c2w = relative_pose(gt_c2w, mode="left")

            img_dir = f"{tmp_dir}/img_{i}"
            os.makedirs(img_dir, exist_ok=True)
            get_frames(str(video_file), img_dir)

            start = time.perf_counter()

            camera_intrinsics = cam_data_gt["camera_intrinsics"]
            fx, fy, cx, cy = camera_intrinsics[0, 0, 0], camera_intrinsics[0, 1, 1], camera_intrinsics[0, 0, 2], camera_intrinsics[0, 1, 2]
            trial_rot_err = []
            trial_trans_err = []
            trial_cam_mc = []
            for _ in range(trials_per_video):
                sample_rel_c2w = compute_camera_poses(img_dir, f"{tmp_dir}/pose_{i}", fx, cx, cy, use_colmap=use_colmap, suppress_output=True)
                if sample_rel_c2w is None:
                    continue
                num_gen_imgs = sample_rel_c2w.shape[0]
                gt_rel_c2w = relative_pose(gt_c2w[:num_gen_imgs], mode="left")
                rot_err, trans_err, cam_mc = metric(gt_rel_c2w.float().clone(), sample_rel_c2w.float().clone())
                trial_rot_err.append(rot_err)
                trial_trans_err.append(trans_err)
                trial_cam_mc.append(cam_mc)
            if len(trial_cam_mc) == 0:
                continue
            elif len(trial_cam_mc) == 1:
                rot_err, trans_err, cam_mc = trial_rot_err[0], trial_trans_err[0], trial_cam_mc[0]
            else:
                if trial_strategy == "average":
                    rot_err, trans_err, cam_mc = np.mean(trial_rot_err), np.mean(trial_trans_err), np.mean(trial_cam_mc)
                elif trial_strategy == "best":
                    min_index = np.argmin(cam_mc)
                    rot_err, trans_err, cam_mc = trial_rot_err[min_index], trial_trans_err[min_index], trial_cam_mc[min_index]
                else:
                    raise ValueError(f"Invalid trial strategy: {trial_strategy}")

            end = time.perf_counter()

            save_dict[str(Path(p).stem)] = {
                "RotErr": rot_err,
                "TransErr": trans_err,
                "CamMC": cam_mc,
                "Time": end - start
            }
            rot_err_list.append(rot_err)
            trans_err_list.append(trans_err)
            cam_mc_list.append(cam_mc)
        except Exception as e:
            print(colored(f"Thread {thread_id}: Error processing '{str(Path(p).stem)}': {str(e)}", "red"))
            traceback.print_exc()

    print(colored(f"Thread {thread_id}: Camera pose evaluation finished!", "green"))
    return save_dict, rot_err_list, trans_err_list, cam_mc_list

def compute_camera_poses(img_dir: str, pose_dir: str, f: float, cx: float, cy: float, use_colmap=False, suppress_output: bool = False) -> tuple:
    def convert(config: dict) -> list[str]:
        return sum([[f"--{k}", f"{v}"] for k, v in config.items()], [])
    
    def run_command(cmd):
        if not suppress_output:
            print(colored("Running: ", "yellow") + " ".join(cmd))
        result = subprocess.run(cmd, shell=False)
        if result.returncode != 0:
            error_message = result.stderr.decode("utf-8")
            print(colored("Error:", "red"), error_message)
            return False
        return True

    model_dir = f"{pose_dir}/model"
    os.makedirs(model_dir, exist_ok=True)

    db_path = f"{pose_dir}/database.db"

    if os.path.exists(db_path):
        os.remove(db_path)

    config = {
        "feature_extractor": {
            "database_path": db_path,
            "image_path": img_dir,
            "ImageReader.single_camera": 1,
            "ImageReader.camera_model": "SIMPLE_PINHOLE",
            "ImageReader.camera_params": f"{f},{cx},{cy}",
            #"SiftExtraction.estimate_affine_shape": 1,
            #"SiftExtraction.domain_size_pooling": 1,
        },
        "sequential_matcher": {
            "database_path": db_path,
            "SiftMatching.guided_matching": 1,
            "SiftMatching.max_num_matches": 65536,
            "SiftMatching.use_gpu": 1
        },
        "mapper": {
            "database_path": db_path,
            "image_path": img_dir,
            "output_path": model_dir,
            "Mapper.init_min_num_inliers": 20
            #"output_format": "txt",
            #"RelPoseEstimation.max_epipolar_error": 4,
            #"BundleAdjustment.optimize_intrinsics": 0,
        },
    }

    if not run_command(["colmap", "feature_extractor"] + convert(config["feature_extractor"])):
        return None
    if not run_command(["colmap", "sequential_matcher"] + convert(config["sequential_matcher"])):
        return None
    
    if not run_command(["glomap" if not use_colmap else "colmap", "mapper"] + convert(config["mapper"])):
        return None

    import ipdb; ipdb.set_trace()
    write_depth_pose_from_colmap_format(f"{model_dir}/0", model_dir, ext=".txt")

    w2c = rt34_to_44(get_rt(f"{model_dir}/poses"))
    c2w = w2c.inverse()
    rel_c2w = relative_pose(c2w, mode="left")

    return rel_c2w


def calc_roterr(r1: Tensor, r2: Tensor) -> Tensor:  # N, 3, 3
    return (((r1.transpose(-1, -2) @ r2).diagonal(dim1=-1, dim2=-2).sum(-1) - 1) / 2).clamp(-1, 1).acos()


def calc_transerr(t1: Tensor, t2: Tensor) -> Tensor:  # N, 3
    return (t2 - t1).norm(p=2, dim=-1)


def calc_cammc(rt1: Tensor, rt2: Tensor) -> Tensor:  # N, 3, 4
    return (rt2 - rt1).reshape(-1, 12).norm(p=2, dim=-1)

def metric(c2w_1: Tensor, c2w_2: Tensor) -> tuple[float, float, float]:  # N, 3, 4
    RotErr = calc_roterr(c2w_1[:, :3, :3], c2w_2[:, :3, :3]).sum().item()  # N, 3, 3

    c2w_1_rel = normalize_t(c2w_1, c2w_1)
    c2w_2_rel = normalize_t(c2w_2, c2w_2)

    TransErr = calc_transerr(c2w_1_rel[:, :3, 3], c2w_2_rel[:, :3, 3]).sum().item()  # N, 3, 1
    CamMC = calc_cammc(c2w_1_rel[:, :3, :4], c2w_2_rel[:, :3, :4]).sum().item()  # N, 3, 4

    return RotErr, TransErr, CamMC




###########################################################
################### Utility Functions #####################
###########################################################

def get_frames(file: str, output_dir: str, ex: bool = False) -> tuple[int, int]:
    mp4 = VideoFileClip(file, audio=False)
    width, height = mp4.size
    align = {1088: 1080, 368: 360}
    if height in align:
        height = align[height]
        mp4 = VideoFileClip(file, audio=False, target_resolution=(height, width))
    # print(round(mp4.fps * mp4.duration), *mp4.size)

    os.makedirs(output_dir, exist_ok=True)
    for idx, frame in enumerate(mp4.iter_frames(logger=None)):
        name = f"{idx:04d}" if ex else f"{idx:03d}"
        imageio.imwrite(f"{output_dir}/{name}.png", frame)
    mp4.close()

    return width, height


def rt34_to_44(rt: Tensor) -> Tensor:
    dummy = torch.tensor([[[0, 0, 0, 1]]] * rt.shape[0], dtype=rt.dtype, device=rt.device)
    return torch.cat([rt, dummy], dim=1)


def relative_pose(rt: Tensor, mode: Literal["left", "right"]) -> Tensor:
    if mode == "left":
        rt = torch.cat([torch.eye(4).unsqueeze(0), rt[:1].inverse() @ rt[1:]], dim=0)
    elif mode == "right":
        rt = torch.cat([torch.eye(4).unsqueeze(0), rt[1:] @ rt[:1].inverse()], dim=0)
    return rt


def normalize_t(rt: Tensor, ref: Tensor = None, eps: float = 1e-9):
    if ref is None:
        ref = rt
    scale = ref[:, :3, 3:4].norm(p=2, dim=1).amax() + eps
    return rt34_to_44(torch.cat([rt[:, :3, :3], rt[:, :3, 3:4] / scale], dim=-1))


def save_depth_pose(output_dir, cameras, images, points3D):
    # Save the sparse depth image and camera pose (world-to-cam) from colmap outputs
    depth_dir = os.path.join(output_dir, "depths")
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    pose_dir = os.path.join(output_dir, "poses")
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)
    intrinsic_dir = os.path.join(output_dir, "intrinsics")
    if not os.path.exists(intrinsic_dir):
        os.makedirs(intrinsic_dir)
    for key in tqdm(images.keys()):
        image_name = images[key].name
        # get camera intrinsics
        camera_id = images[key].camera_id
        camera = cameras[camera_id]
        h, w, params = camera.height, camera.width, camera.params
        if camera.model == 'SIMPLE_PINHOLE':
            f, cx, cy = params
        elif camera.model == 'SIMPLE_RADIAL':
            f, cx, cy, d = params
        else:
            raise NotImplementedError
        K = np.array([[f, 0, cx], [0, f, cy], [0,0,1]])
        np.savetxt(os.path.join(intrinsic_dir, os.path.splitext(image_name)[0]+'.txt'), K)
        # world-to-cam quaternion and translation
        qvec, tvec = images[key].qvec, images[key].tvec
        R, t = qvec2rotmat(qvec), np.expand_dims(tvec, -1)
        # acquire 3d points
        xys = images[key].xys
        points3D_ids = images[key].point3D_ids
        points_3d, valid_xys = [], []
        for i in range(len(points3D_ids)):
            idx = points3D_ids[i]
            if idx == -1:
                continue
            points_3d.append((points3D[idx].xyz))
            valid_xys.append(xys[i])

        points_3d = np.transpose(np.array(points_3d))
        # project onto image
        cam_points = np.matmul(R, points_3d) + t
        project_depth = np.transpose(np.matmul(K, cam_points))[:,-1]
        xy_int = np.round(np.array(valid_xys)).astype(np.int32)
        xy_int[:,0] = np.clip(xy_int[:,0], 0, w-1)
        xy_int[:,1] = np.clip(xy_int[:,1], 0, h-1)
        depth = np.zeros(shape=(h,w))
        depth[xy_int[:,1], xy_int[:,0]] = project_depth
        # save depth and pose
        np.save(os.path.join(depth_dir, os.path.splitext(image_name)[0]+'.npy'), depth)
        plt.imsave(os.path.join(depth_dir, os.path.splitext(image_name)[0]+'.png'), normalize_depth_for_display(depth, cmap='binary'))
        np.savetxt(os.path.join(pose_dir, os.path.splitext(image_name)[0]+'.txt'), np.concatenate([R, t], -1))

def normalize_depth_for_display(depth, pc=98, cmap='gray', eps=1e-9):
    vinds = depth > 0
    depth = 1. / (depth + 1)
    z1 = np.percentile(depth[vinds], pc)
    z2 = np.percentile(depth[vinds], 100-pc)

    depth = (depth - z2) / (z1 - z2 + eps)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    return depth

def gray2rgb(im, cmap):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def write_depth_pose_from_colmap_format(input_dir, output_dir, ext=''):
    model = read_model(input_dir, ext=ext)
    if model is None:
        return
    else:
        cameras, images, points3D = model
    save_depth_pose(output_dir, cameras, images, points3D)


def get_rt(folder: str) -> Tensor:
    files = sorted([x for x in glob(f"{folder}/*.txt") if re.search(r"(\d+)\.txt$", x)])
    return torch.stack([load_rt_from_txt(file) for file in files])


def load_rt_from_txt(file_path: str, comments: str = None) -> Tensor:
    return torch.from_numpy(np.loadtxt(file_path, comments=comments, dtype=np.float64))



def read_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            try:
                cameras, images, points3D = read_model(os.path.join(path, "model/"))
                logging.warning(
                    "This SfM file structure was deprecated in hloc v1.1")
                return cameras, images, points3D
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find binary or text COLMAP model at {path}")

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def write_model(cameras, images, points3D, path, ext=".bin"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        write_images_binary(images, os.path.join(path, "images" + ext))
        write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
    return cameras, images, points3D




def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = "# Camera list with one line of data per camera:\n" + \
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + \
             "# Number of cameras: {}\n".format(len(cameras))
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id,
                                 model_id,
                                 cam.width,
                                 cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum((len(img.point3D_ids) for _, img in images.items()))/len(images)
    HEADER = "# Image list with two lines of data per image:\n" + \
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n" + \
             "# Number of images: {}, mean observations per image: {}\n".format(len(images), mean_observations)

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items()))/len(points3D)
    HEADER = "# 3D point list with one line of data per point:\n" + \
             "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n" + \
             "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def detect_model_format(path, ext):
    if os.path.isfile(os.path.join(path, "cameras"  + ext)) and \
       os.path.isfile(os.path.join(path, "images"   + ext)) and \
       os.path.isfile(os.path.join(path, "points3D" + ext)):
        return True

    return False


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


###########################################################
########################## Main ###########################
###########################################################


def main(path: Path, num_workers: int = 1):

    video_paths = [p for p in sorted(path.iterdir()) if p.is_dir()]
    valid_paths = []

    for p in video_paths:
        gt_path = p / "raw" / "ground_truth.mp4"
        v_path = p / "raw" / "generated.mp4"

        if gt_path.exists() and v_path.exists():
            valid_paths.append(str(p))

    print(f"Found {len(valid_paths)} valid videos for evaluation.")
    print(f"Valid videos: {valid_paths}")
    
    if len(valid_paths) == 0:
        print("No valid videos found for evaluation.")
        return
    
    # Split valid_paths into equal parts for each worker
    chunk_size = math.ceil(len(valid_paths) / num_workers)
    path_chunks = [valid_paths[i:i + chunk_size] for i in range(0, len(valid_paths), chunk_size)]
    
    print(f"Splitting {len(valid_paths)} videos into {len(path_chunks)} chunks for {num_workers} workers")
    for i, chunk in enumerate(path_chunks):
        print(f"Chunk {i}: {len(chunk)} videos")
    
    # Create output directory
    output_dir = path / "camera_evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    # Use ThreadPoolExecutor to process chunks in parallel
    all_save_dicts = {}
    all_rot_err = []
    all_trans_err = []
    all_cam_mc = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks for each chunk
        futures = []
        for i, chunk in enumerate(path_chunks):
            if len(chunk) > 0:  # Only submit non-empty chunks
                future = executor.submit(camera_pose_evaluation, chunk, output_dir, i)
                futures.append(future)
        
        # Collect results from all threads
        for future in futures:
            try:
                save_dict, rot_err_list, trans_err_list, cam_mc_list = future.result()
                
                # Combine results
                all_save_dicts.update(save_dict)
                all_rot_err.extend(rot_err_list)
                all_trans_err.extend(trans_err_list)
                all_cam_mc.extend(cam_mc_list)
                
            except Exception as e:
                print(colored(f"Error in thread execution: {str(e)}", "red"))
                traceback.print_exc()
    
    # Save combined results
    if all_save_dicts:
        detail_eval_file = output_dir / "camera_eval.yaml"
        with open(detail_eval_file, "w") as file:
            yaml.dump(all_save_dicts, file, default_flow_style=False, sort_keys=False)
        
        # Calculate and print overall statistics
        mean_rot_err = np.mean(all_rot_err) if all_rot_err else 0
        mean_trans_err = np.mean(all_trans_err) if all_trans_err else 0
        mean_cam_mc = np.mean(all_cam_mc) if all_cam_mc else 0
        
        print(colored("=== FINAL RESULTS ===", "green"))
        print(f"Total videos processed: {len(all_save_dicts)}")
        print(f"Mean Rotation Error: {mean_rot_err:.6f}")
        print(f"Mean Translation Error: {mean_trans_err:.6f}")
        print(f"Mean Camera Motion Consistency: {mean_cam_mc:.6f}")
        
        # Save summary statistics
        summary_stats = {
            "total_videos": len(all_save_dicts),
            "mean_rotation_error": float(mean_rot_err),
            "mean_translation_error": float(mean_trans_err), 
            "mean_camera_motion_consistency": float(mean_cam_mc),
            "num_workers": num_workers
        }
        
        summary_file = output_dir / "summary_stats.yaml"
        with open(summary_file, "w") as file:
            yaml.dump(summary_stats, file, default_flow_style=False)
            
        print(f"Results saved to: {detail_eval_file}")
        print(f"Summary saved to: {summary_file}")
    else:
        print(colored("No results to save - all evaluations failed.", "red"))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=Path, required=True, help="Path to the directory containing the evaluation results.")
    parser.add_argument("--num_workers", "-n", type=int, default=1, help="Number of worker threads to use for parallel processing.")
    args = parser.parse_args()

    main(**vars(args))