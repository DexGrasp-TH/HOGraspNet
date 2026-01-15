import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set the environment variable
current_abs_path = os.getcwd()
os.environ["HOG_DIR"] = current_abs_path

# 确保路径正确
sys.path.append(".")
if "HOG_DIR" in os.environ:
    sys.path.append(os.environ["HOG_DIR"])
    sys.path.append(os.path.join(os.environ["HOG_DIR"], "thirdparty/manopth"))

# 假设 HOG_dataloader 等已正确配置
from HOG_dataloader import HOGDataset
from config import cfg
from thirdparty.manopth.manopth.manolayer import ManoLayer
from pytorch3d.io import load_obj

# 定义手指末端的索引 (根据你提供的图片)
# 4: Thumb, 8: Index, 12: Middle, 16: Ring, 20: Pinky
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def get_diagonal_scale(verts):
    """计算物体的对角线跨度"""
    mins = torch.min(verts, dim=0)[0]
    maxs = torch.max(verts, dim=0)[0]
    return torch.norm(maxs - mins).item()


def load_object_meshes(model_path, device="cpu"):
    obj_templates = {}
    obj_templates["verts_h"] = {}
    obj_templates["faces"] = {}

    obj_list = os.listdir(model_path)

    print("loading object meshes ...")
    for obj_name in obj_list:
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
    print("... done")

    return obj_templates


if __name__ == "__main__":
    setup = "s5"
    split = "test"
    db_path = os.environ["HOG_DIR"] + "/data"

    HOG = HOGDataset(setup, split, db_path=db_path)
    HOG_loader = DataLoader(HOG, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 MANO
    mano_layer = ManoLayer(
        side="right",
        mano_root=cfg.mano_path,
        use_pca=False,
        flat_hand_mean=True,
        center_idx=0,
        ncomps=45,
        root_rot_mode="axisang",
        joint_rot_mode="axisang",
    ).to(device)

    obj_model_path = os.path.join(db_path, "obj_scanned_models")
    obj_templates = load_object_meshes(model_path=obj_model_path, device=device)

    # 统计字典：键为 scale_bin，值为各个手指的计数
    stats = {}
    # 接触阈值：指尖距离物体表面 < 2cm (0.02m) 认为在抓取
    CONTACT_THRESHOLD = 2  # cm

    print("Processing dataset...")
    for idx, sample in enumerate(tqdm(HOG_loader)):
        # --- 1. 获取物体顶点并计算 Scale ---
        anno = sample["anno_data"]
        obj_id = int(sample["obj_ids"][0])
        obj_verts_template_h = obj_templates["verts_h"][obj_id]
        obj_faces_template = obj_templates["faces"][obj_id]

        # 加载物体模板（简化处理，直接从anno获取变换后的顶点）
        obj_mat = torch.FloatTensor(anno["Mesh"][0]["object_mat"]).to(device)
        obj_points = obj_verts_template_h @ obj_mat.T
        obj_verts_world = obj_points[:, :3] / obj_points[:, 3:]
        obj_verts_world = obj_verts_world.view(-1, 3)

        diag_scale = get_diagonal_scale(obj_verts_world)

        # 将 scale 分组
        bin_size = 5  # cm
        scale_bin = (diag_scale // bin_size) * bin_size

        if scale_bin not in stats:
            stats[scale_bin] = {name: 0 for name in FINGER_NAMES}
            stats[scale_bin]["total_samples"] = 0
            stats[scale_bin]["invalid_samples"] = 0

        # --- 2. 获取手部顶点/关节点 ---
        hand_mano_rot = torch.FloatTensor(anno["Mesh"][0]["mano_trans"]).to(device)
        hand_mano_pose = torch.FloatTensor(anno["Mesh"][0]["mano_pose"]).to(device)
        hand_mano_shape = torch.FloatTensor(anno["Mesh"][0]["mano_betas"]).to(device)

        mano_param = torch.cat([hand_mano_rot, hand_mano_pose], dim=1)
        _, mano_joints = mano_layer(mano_param, hand_mano_shape)

        # 转换到世界坐标 (参考你原代码的 root_trans 和 scale 处理)
        hand_scale = anno["hand"]["mano_scale"].to(device)
        hand_xyz_root = torch.Tensor(anno["hand"]["mano_xyz_root"]).to(device)
        mano_joints_world = (mano_joints[0] / hand_scale) + hand_xyz_root

        # TODO: use hand joints and compare

        # --- 3. 接触检测 ---
        n_contact = 0
        tip_dists = {name: 0 for name in FINGER_NAMES}
        for f_idx, tip_id in enumerate(FINGER_TIPS):
            tip_pos = mano_joints_world[tip_id]  # (3,)

            # 计算指尖到物体所有顶点的最小距离
            dists = torch.norm(obj_verts_world - tip_pos, dim=1)
            min_dist = torch.min(dists).item()
            tip_dists[FINGER_NAMES[f_idx]] = min_dist
            if min_dist < CONTACT_THRESHOLD:
                n_contact += 1

        if n_contact >= 2:  # a valid grasp contains at least two contacted fingertips
            stats[scale_bin]["total_samples"] += 1
            for f_idx, tip_id in enumerate(FINGER_TIPS):
                if tip_dists[FINGER_NAMES[f_idx]] < CONTACT_THRESHOLD:
                    stats[scale_bin][FINGER_NAMES[f_idx]] += 1
        else:
            stats[scale_bin]["invalid_samples"] += 1
            print(f"Invalid label_path: {sample['label_path']}")

    # --- 4. 打印与可视化 ---
    print("\nScale (cm) | Count | Thumb | Index | Middle | Ring | Pinky | Invalid")
    sorted_bins = sorted(stats.keys())
    for b in sorted_bins:
        total = stats[b]["total_samples"]
        invalid = stats[b]["invalid_samples"]

        # 如果该 bin 下没有样本（理论上不会，但增加健壮性），跳过
        if total == 0:
            continue

        # 计算每个手指的使用比例
        row = [f"{stats[b][name] / total:.2f}" for name in FINGER_NAMES]

        # 打印：Scale值 | 样本总数 | 各个手指分布
        # {b:10.1f} 表示占据10个字符位，保留1位小数
        # {total:5d} 表示占据5个字符位，整数显示
        print(
            f"{b:<10.1f} | {total:<5d} | " + "  | ".join(row) + f" | {invalid:<10.1f}"
        )
