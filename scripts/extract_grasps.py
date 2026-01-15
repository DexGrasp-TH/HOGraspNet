"""HOGraspNet dataset."""

import os
import sys
import shutil
import trimesh
import re

# Set the environment variable
current_abs_path = os.getcwd()
os.environ["HOG_DIR"] = current_abs_path

sys.path.append(".")
sys.path.append(os.environ["HOG_DIR"])
sys.path.append(os.path.join(os.environ["HOG_DIR"], "thirdparty/manopth"))

import json
import torch
import numpy as np
from tqdm import tqdm
from config import cfg
from pytorch3d.io import load_obj

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
NEW_DATASET_PATH = "/data/dataset/AnyScaleGrasp/HOGraspNet"


def load_object_meshes(model_path, device="cpu"):
    obj_templates = {}
    obj_templates["verts_h"] = {}
    obj_templates["faces"] = {}

    obj_list = os.listdir(model_path)

    print("loading object meshes ...")
    for obj_name in tqdm(obj_list):
        obj_mesh_path = os.path.join(model_path, obj_name, obj_name + ".obj")
        obj_idx = int(obj_name.split("_")[0])
        obj_scale = cfg._OBJECT_SCALE_FIXED[obj_idx - 1]

        obj_verts, obj_faces, _ = load_obj(obj_mesh_path)
        obj_verts_template = (obj_verts * float(obj_scale)).to(device)
        obj_faces_template = torch.unsqueeze(obj_faces.verts_idx, axis=0).to(device)

        h = torch.ones((obj_verts_template.shape[0], 1), device=device)
        obj_verts_template_h = torch.cat((obj_verts_template, h), 1)

        obj_templates["verts_h"][obj_idx] = obj_verts_template_h
        obj_templates["faces"][obj_idx] = obj_faces_template

        # ########### Load textured object mesh using trimesh and re-save it ###########
        # # Load (trimesh automatically looks for the .mtl and image files)
        # mesh = trimesh.load(obj_mesh_path, force="mesh")

        # save_dir = os.path.join(NEW_DATASET_PATH, "object", obj_name)
        # os.makedirs(save_dir, exist_ok=True)

        # # When exporting textured OBJs, trimesh creates a .obj, a .mtl, and the image file
        # mesh.export(os.path.join(save_dir, f"{obj_name}.obj"))
        # print(f"Successfully exported {obj_name} and its textures to: {save_dir}")

    print("... done")

    return obj_templates


class HOGDataset:
    def __init__(self, db_path, verbose=False):
        """Constructor."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        self._base_dir = db_path
        self._base_anno = os.path.join(self._base_dir, "labeling_data")
        self._base_source = os.path.join(self._base_dir, "source_data")
        self._base_source_aug = os.path.join(self._base_dir, "source_augmented")

        self._base_extra = os.path.join(self._base_dir, "extra_data")
        self._obj_model_dir = os.path.join(self._base_dir, "obj_scanned_models")

        self._h = 480
        self._w = 640

        self.camIDset = cfg._CAMIDSET

        ## MINING SEQUENCE INFOS
        self._SUBJECTS, self._OBJ_IDX, self._GRASP_IDX, self._OBJ_GRASP_PAIR = (
            [],
            [],
            [],
            [],
        )

        seq_list = sorted(os.listdir(self._base_anno))
        self._seq_dict_list = []
        for idx, seq in enumerate(seq_list):
            seq_info = {}
            seq_split = seq.split("_")

            seq_info["idx"] = idx
            seq_info["seqName"] = seq
            # seq_info['date'] = seq_split[0]
            seq_info["subject"] = seq_split[1]
            seq_info["obj_idx"] = seq_split[3]
            seq_info["grasp_idx"] = seq_split[5]
            seq_info["obj_grasp_pair"] = [seq_split[3], seq_split[5]]

            if seq_info["subject"] not in self._SUBJECTS:
                self._SUBJECTS.append(seq_info["subject"])

            if seq_info["obj_idx"] not in self._OBJ_IDX:
                self._OBJ_IDX.append(seq_info["obj_idx"])

            if seq_info["grasp_idx"] not in self._GRASP_IDX:
                self._GRASP_IDX.append(seq_info["grasp_idx"])

            if seq_info["obj_grasp_pair"] not in self._OBJ_GRASP_PAIR:
                self._OBJ_GRASP_PAIR.append(seq_info["obj_grasp_pair"])

            self._seq_dict_list.append(seq_info)

        ## TRAIN / TEST / VALID SPLIT
        subject_ind = "S01"
        serial_ind = self.camIDset
        obj_grasp_pair_ind = self._OBJ_GRASP_PAIR
        trial_ind = "full"  # 'full', 'train', 'val', 'test'

        obj_model_path = os.path.join(db_path, "obj_scanned_models")
        obj_templates = load_object_meshes(model_path=obj_model_path, device=self.device)

        #########################################

        # for each object has its mapping index which contains s,t,c,f (subject,trial,cam,frame)
        total_grasp_account = 0
        saved_grasp_account = 0
        self.load = False
        self.mapping = []  # its location
        self.cam_param_dict = {}

        for seqIdx, seq in enumerate(tqdm(self._seq_dict_list)):
            # skip if not target sequence
            if seq["subject"] not in subject_ind:
                continue
            if seq["obj_grasp_pair"] not in obj_grasp_pair_ind:
                continue

            seqName = seq["seqName"]
            seqDir = os.path.join(self._base_anno, seqName)

            # get object template
            obj_id = int(seq["obj_idx"])
            obj_verts_template_h = obj_templates["verts_h"][obj_id]
            obj_faces_template = obj_templates["faces"][obj_id]

            for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
                # skip if not target trial
                if trial_ind == "train" and trialIdx == 0:
                    continue
                if trial_ind == "test" and trialIdx != 0:
                    continue
                if trial_ind == "valid" and trialIdx != 1:
                    continue

                anno_base_path = os.path.join(seqDir, trialName, "annotation")
                valid_cams = os.listdir(anno_base_path)

                if self.verbose:
                    print(f"anno_base_path: {anno_base_path}")

                self.anno_dict = self.load_data(seqName, trialName, valid_cams)
                total_grasp_account += 1

                ################## 在所有相机中寻找最优抓取帧 ##################
                motion_threshold = 1  # cm
                CONTACT_THRESHOLD = 2  # cm

                best_overall_anno = None
                best_overall_n_contact = -1
                best_overall_mean_dist = float("inf")
                best_overall_camID = None
                best_overall_tip_dists = None

                for camID in valid_cams:
                    if camID not in serial_ind or camID not in self.anno_dict:
                        continue

                    target_annos = self.anno_dict[camID]
                    init_obj_mat = None
                    cam_select_anno_idx = 0

                    # 1. 在当前相机序列中寻找第一个发生运动的帧
                    for anno_idx, anno_path in enumerate(target_annos):
                        with open(anno_path, "r", encoding="UTF-8 SIG") as file:
                            temp_anno = json.load(file)

                        current_obj_mat = torch.FloatTensor(temp_anno["Mesh"][0]["object_mat"]).to(self.device)

                        if init_obj_mat is not None:
                            # 计算位移 (cm)
                            mat_diff = torch.norm(current_obj_mat[:3, 3] - init_obj_mat[:3, 3]).item()

                            if mat_diff > motion_threshold:
                                # 找到该相机下的运动起始帧，记录并跳出该相机的循环
                                cam_select_anno_idx = anno_idx - 1
                                if self.verbose:
                                    print(
                                        f"Select anno_idx={cam_select_anno_idx} as the motion frame of camera {camID}."
                                    )
                                break

                        if anno_idx == 0:
                            init_obj_mat = current_obj_mat.clone()

                    cam_best_path = target_annos[cam_select_anno_idx]
                    with open(anno_path, "r", encoding="UTF-8 SIG") as file:
                        cam_best_anno = json.load(file)

                    # 2. 如果该相机找到了运动帧，计算其抓取质量
                    if cam_best_anno is not None:
                        obj_mat = torch.FloatTensor(cam_best_anno["Mesh"][0]["object_mat"]).to(self.device)
                        obj_points = obj_verts_template_h @ obj_mat.T
                        obj_verts_world = obj_points[:, :3] / obj_points[:, 3:]
                        obj_verts_world = obj_verts_world.view(-1, 3)

                        hand_joints_3d = torch.tensor(cam_best_anno["hand"]["3D_pose_per_cam"], device=self.device)

                        temp_tip_dists = torch.zeros((len(FINGER_TIPS)), device=self.device)
                        for f_idx, tip_id in enumerate(FINGER_TIPS):
                            tip_pos = hand_joints_3d[tip_id]
                            dists = torch.norm(obj_verts_world - tip_pos, dim=1)
                            temp_tip_dists[f_idx] = torch.min(dists).item()

                        # 计算接触数量和被判定为接触的指尖平均距离
                        contact_mask = temp_tip_dists < CONTACT_THRESHOLD
                        n_contact = contact_mask.sum().item()

                        if n_contact >= 2:
                            mean_dist = temp_tip_dists[contact_mask].mean().item()

                            # 3. 跨相机择优逻辑：
                            # 优先比接触点数量 (n_contact)
                            # 数量相同时，比平均接触距离 (mean_dist)
                            if (n_contact > best_overall_n_contact) or (
                                n_contact == best_overall_n_contact and mean_dist < best_overall_mean_dist
                            ):
                                best_overall_n_contact = n_contact
                                best_overall_mean_dist = mean_dist
                                best_overall_anno = cam_best_anno
                                best_overall_camID = camID
                                best_overall_tip_dists = temp_tip_dists
                                select_anno_path = cam_best_path

                # 检查最终是否找到了符合要求的抓取
                if best_overall_anno is None:
                    if self.verbose:
                        print(f"[{seqName}/{trialName}] No valid grasp (n_contact >= 2) found across all cameras.")
                    continue

                # 将选定的最优数据赋值给 anno 供后续保存使用
                anno = best_overall_anno
                tip_dists = best_overall_tip_dists
                if self.verbose:
                    print(
                        f"Best Grasp Found: Cam {best_overall_camID} | Contacts: {best_overall_n_contact} | Mean Dist: {best_overall_mean_dist:.4f}"
                    )

                ################## Re-format and save the dataset ##################
                new_data = {
                    "object": {},
                    "hand": {
                        "left": {},
                        "right": {},
                    },
                    "extra": {},
                }

                new_data["object"]["name"] = obj_name = f"{obj_id:02d}_{anno['Mesh'][0]['object_name']}"
                new_data["object"]["path"] = os.path.join(NEW_DATASET_PATH, "object", obj_name, f"{obj_name}.obj")
                new_data["object"]["rel_scale"] = cfg._OBJECT_SCALE_FIXED[obj_id - 1] / 100.0  # unit cm to m
                new_data["object"]["pose"] = np.asarray(anno["Mesh"][0]["object_mat"])  # 4*4 matrix
                new_data["object"]["pose"][:3, 3] = new_data["object"]["pose"][:3, 3] / 100.0  # unit cm to m

                side = anno["Mesh"][0]["mano_side"]
                new_data["hand"][side]["scale"] = anno["hand"]["mano_scale"] * 100.0  # unit cm to m
                new_data["hand"][side]["trans"] = np.asarray(anno["hand"]["mano_xyz_root"]) / 100.0  # unit cm to m
                new_data["hand"][side]["rot"] = anno["Mesh"][0]["mano_trans"]
                new_data["hand"][side]["mano_pose"] = anno["Mesh"][0]["mano_pose"]
                new_data["hand"][side]["mano_betas"] = anno["Mesh"][0]["mano_betas"]
                new_data["hand"][side]["contacts"] = (tip_dists < CONTACT_THRESHOLD).cpu().tolist()

                new_data["extra"]["select_anno_path"] = select_anno_path

                save_dir = os.path.join(NEW_DATASET_PATH, "raw_grasp", seqName)
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, f"{trialName}.npy")
                np.save(file_path, new_data)
                if self.verbose:
                    print(f"Successfully saved grasp: {seqName}/{trialName}.npy | Side: {side} | Object: {obj_name}")

                saved_grasp_account += 1

        print(f"Total grasp account: {total_grasp_account}, saved_grasp account: {saved_grasp_account}.")

    def load_data(self, seqName, trialName, valid_cams):
        anno_base_path = os.path.join(self._base_anno, seqName, trialName, "annotation")
        anno_dict = {}

        def sort_key_func(filename):
            # 匹配文件名末尾的数字（例如从 "xxx_12.jpg" 中提取 "12"）
            match = re.search(r"_(\d+)\.(jpg|json|npy)", filename)
            if match:
                return int(match.group(1))
            return 0

        for camIdx, camID in enumerate(self.camIDset):
            anno_dict[camID] = []

            if camID in valid_cams:
                anno_list = os.listdir(os.path.join(anno_base_path, camID))

                sorted_anno_list = sorted(anno_list, key=sort_key_func)

                for anno in sorted_anno_list:
                    anno_path = os.path.join(anno_base_path, camID, anno)
                    anno_dict[camID].append(anno_path)

        return anno_dict


def main():
    db_path = os.path.join(os.environ["HOG_DIR"], "data")
    HOG = HOGDataset(db_path, verbose=True)


if __name__ == "__main__":
    main()
