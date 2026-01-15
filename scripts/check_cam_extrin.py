import os
import json
import numpy as np
from tqdm import tqdm
import sys

# Set the environment variable
current_abs_path = os.getcwd()
os.environ["HOG_DIR"] = current_abs_path

sys.path.append(".")
sys.path.append(os.environ["HOG_DIR"])


def check_extrinsic_consistency(db_path):
    """
    Checks if extrinsic matrices for each camera ID remain constant
    across different sequences and trials.
    """
    anno_base_dir = os.path.join(db_path, "labeling_data")

    # Dictionary to store the first seen extrinsic for each camera
    # { 'cam_id': {'matrix': np.array, 'origin': 'seq_name/trial_name'} }
    reference_extrinsics = {}
    inconsistencies = []

    # Get all sequences
    if not os.path.exists(anno_base_dir):
        print(f"Error: {anno_base_dir} not found.")
        return

    subject_filter = "S01"

    # 获取所有序列并筛选以指定前缀开头的文件夹
    all_seqs = sorted(os.listdir(anno_base_dir))
    # 同时满足日期前缀和 S01 过滤条件
    target_seqs = [s for s in all_seqs if subject_filter in s]

    if not target_seqs:
        print(f"未找到满足条件（过滤: {subject_filter}）的序列。")
        return

    print(f"正在分析 {len(target_seqs)} 个序列...")

    for seq_name in tqdm(target_seqs):
        seq_path = os.path.join(anno_base_dir, seq_name)
        if not os.path.isdir(seq_path):
            continue

        for trial_name in os.listdir(seq_path):
            anno_path = os.path.join(seq_path, trial_name, "annotation")
            if not os.path.exists(anno_path):
                continue

            # Check every camera folder in this trial
            for cam_id in os.listdir(anno_path):
                cam_dir = os.path.join(anno_path, cam_id)
                files = os.listdir(cam_dir)
                if not files:
                    continue

                # We only need to check one frame per trial/cam to see if the setup moved
                sample_json = os.path.join(cam_dir, files[0])

                with open(sample_json, "r", encoding="UTF-8 SIG") as f:
                    data = json.load(f)

                # Extract extrinsic matrix
                ms = np.squeeze(np.asarray(data["calibration"]["extrinsic"]))
                ms = np.reshape(ms, (3, 4))

                if cam_id not in reference_extrinsics:
                    # Store the first one we find as the reference
                    reference_extrinsics[cam_id] = {"matrix": ms, "origin": f"{seq_name}/{trial_name}"}
                else:
                    # Compare with reference using a small tolerance for floating point errors
                    ref_ms = reference_extrinsics[cam_id]["matrix"]
                    if not np.allclose(ms, ref_ms, atol=1e-1):
                        diff = np.abs(ms - ref_ms).max()
                        inconsistencies.append(
                            {
                                "cam_id": cam_id,
                                "ref_origin": reference_extrinsics[cam_id]["origin"],
                                "curr_origin": f"{seq_name}/{trial_name}",
                                "max_diff": diff,
                            }
                        )

    # --- Report Results ---
    if not inconsistencies:
        print("\n✅ Success: Extrinsics are CONSTANT for all cameras across the dataset.")
    else:
        print(f"\n❌ Found {len(inconsistencies)} inconsistent extrinsic instances!")
        for inc in inconsistencies[:10]:  # Print first 10
            print(
                f"Cam {inc['cam_id']}: Changed in {inc['curr_origin']} "
                f"(Max Delta vs {inc['ref_origin']}: {inc['max_diff']:.6f})"
            )
        if len(inconsistencies) > 10:
            print("...")


if __name__ == "__main__":
    # Ensure HOG_DIR is set in your environment or replace with direct path
    database_path = os.environ.get("HOG_DIR", "./")
    database_path = os.path.join(database_path, "data")

    check_extrinsic_consistency(database_path)
