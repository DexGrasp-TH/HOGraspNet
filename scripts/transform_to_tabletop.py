import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as sciR

# TRANS_C_IN_W = [
#     [0.74924768, -0.42582172, 0.5072512, -0.50848046],
#     [-0.6525146, -0.60572749, 0.45532297, -0.37870704],
#     [0.11336964, -0.67213841, -0.73169481, 0.56076683],
#     [0.0, 0.0, 0.0, 1.0],
# ]

# TRANS_C_IN_W = [
#     [-0.0, -0.99720272, 0.07474451, -0.02132014],
#     [0.99983775, -0.00134638, -0.01796264, 0.02282197],
#     [0.01801303, 0.07473238, 0.99704092, -0.03760181],
#     [0.0, 0.0, 0.0, 1.0],
# ]


def main():
    """
    Transform the hand pose and object pose from the MAS camera frame (world frame of HOGraspNet) to the tabletop world frame.
    """

    TRANS_C_IN_W = np.load("data/mas_in_tabletop_frame.npy")

    folder_path = "/data/dataset/AnyScaleGrasp/HOGraspNet/raw_grasp/*/*.npy"
    all_grasps_paths = glob.glob(folder_path)

    for path in all_grasps_paths:
        data = np.load(path, allow_pickle=True).item()

        # Transform the hand base pose
        side = "right" if data["hand"]["right"] else "left"
        hand_info = data["hand"][side]

        if "trans" not in hand_info.keys():
            a = 1

        hand_trans = np.array(hand_info["trans"])
        hand_rot = np.array(hand_info["rot"])

        hand_base_pose = np.eye(4)
        hand_base_pose[:3, 3] = hand_trans
        hand_base_pose[:3, :3] = sciR.from_rotvec(hand_rot[0]).as_matrix()

        hand_base_pose_in_w = TRANS_C_IN_W @ hand_base_pose
        hand_trans = hand_base_pose_in_w[:3, 3]
        hand_rot = sciR.from_matrix(hand_base_pose_in_w[:3, :3]).as_rotvec().reshape(1, 3)
        data["hand"][side]["trans"] = hand_trans
        data["hand"][side]["rot"] = hand_rot

        # Transform the object pose
        obj_pose = np.asarray(data["object"]["pose"])  # 4x4 matrix
        obj_pose = TRANS_C_IN_W @ obj_pose
        data["object"]["pose"] = obj_pose

        # Re-save the data
        save_path = path.replace("/raw_grasp", "/grasp")
        np.save(save_path, data)


if __name__ == "__main__":
    main()
