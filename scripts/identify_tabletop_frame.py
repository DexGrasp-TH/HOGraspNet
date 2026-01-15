import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as sciR
import json
import re


def compute_tabletop_frame(obj_poses):
    """
    obj_poses: (N, 4, 4) list or array
    """
    poses = np.array(obj_poses)

    # 1. 提取所有物体的 Y 轴 (即桌面的法向量候选)
    # 在 4x4 矩阵中，第二列 (index 1) 是 Y 轴向量
    y_axes = poses[:, :3, 1]

    # 2. 计算平均法向量并归一化
    mean_y_axis = np.mean(y_axes, axis=0)
    normal = mean_y_axis / np.linalg.norm(mean_y_axis)

    z_table = normal

    # 3. 确定坐标系原点 (取所有物体位置的中心点)
    centers = poses[:, :3, 3]
    origin = np.mean(centers, axis=0)

    # 4. 构建 X 和 Y 轴
    # 选一个临时的参考向量（不能与 z_table 平行）
    ref_vec = np.array([1, 0, 0]) if abs(z_table[0]) < 0.9 else np.array([0, 1, 0])

    x_table = np.cross(ref_vec, z_table)
    x_table /= np.linalg.norm(x_table)

    y_table = np.cross(z_table, x_table)

    # 5. 组装成 4x4 矩阵
    table_to_cam = np.eye(4)
    table_to_cam[:3, 0] = x_table
    table_to_cam[:3, 1] = y_table
    table_to_cam[:3, 2] = z_table
    table_to_cam[:3, 3] = origin

    return table_to_cam


def sort_key_func(filename):
    # 匹配文件名末尾的数字（例如从 "xxx_12.jpg" 中提取 "12"）
    match = re.search(r"_(\d+)\.(jpg|json|npy)", filename)
    if match:
        return int(match.group(1))
    return 0


def main():
    """
    Transform the hand pose and object pose from the MAS camera frame (world frame of HOGraspNet) to the tabletop world frame.
    """
    all_grasps_paths = [
        "230905_S01_obj_24_grasp_19/trial_0",
        "231023_S01_obj_05_grasp_24/trial_0",
        "231023_S01_obj_07_grasp_31/trial_0",
        "231023_S01_obj_07_grasp_31/trial_1",
        "231023_S01_obj_08_grasp_18/trial_0",
        "231023_S01_obj_08_grasp_30/trial_1",
        "231023_S01_obj_21_grasp_12/trial_0",
        "231023_S01_obj_24_grasp_1/trial_1",
        "231023_S01_obj_30_grasp_28/trial_0",
    ]

    obj_poses = []

    anno_base_dir = os.path.join("data/labeling_data")

    # Get all sequences
    if not os.path.exists(anno_base_dir):
        print(f"Error: {anno_base_dir} not found.")
        return

    # 直接遍历指定的路径列表
    for rel_path in all_grasps_paths:
        seq_name, trial_name = os.path.split(rel_path)

        # 拼接完整的 annotation 目录路径
        # 路径结构: data/labeling_data/seq_name/trial_name/annotation
        anno_dir = os.path.join(anno_base_dir, seq_name, trial_name, "annotation")

        if not os.path.exists(anno_dir):
            continue

        cam_dir = os.path.join(anno_dir, "mas")

        sorted_anno_list = sorted(os.listdir(cam_dir), key=sort_key_func)

        # We only need to check one frame per trial/cam to see if the setup moved
        sample_json = os.path.join(cam_dir, sorted_anno_list[0])

        with open(sample_json, "r", encoding="UTF-8 SIG") as f:
            data = json.load(f)

        obj_pose = np.array(data["Mesh"][0]["object_mat"])

        obj_pose[:3, 3] = obj_pose[:3, 3] / 100.0
        obj_poses.append(obj_pose)

    table_to_cam = compute_tabletop_frame(obj_poses)
    cam_in_table = np.linalg.inv(table_to_cam)
    np.save("data/mas_in_tabletop_frame.npy", cam_in_table)

    print("cam_in_table: ", cam_in_table)


if __name__ == "__main__":
    main()
