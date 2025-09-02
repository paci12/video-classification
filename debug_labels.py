#!/usr/bin/env python3

import os
import pickle
from sklearn.preprocessing import LabelEncoder

# 配置
data_path = "/data2/lpq/video-classification/jpegs_256_processed/"
action_name_path = "data/UCF101actions.pkl"

print("Loading action names...")
with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)

print(f"Action names: {action_names[:10]}")
print(f"Total actions: {len(action_names)}")

# 标签编码器
le = LabelEncoder()
le.fit(action_names)

print("\nGetting all directories...")
dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
print(f"Total directories: {len(dirs)}")
print(f"First 10 directories: {dirs[:10]}")

actions = []
all_names = []

print("\nParsing video files...")
for action_dir in dirs:
    action_dir_path = os.path.join(data_path, action_dir)
    video_files = os.listdir(action_dir_path)
    
    for video_file in video_files:
        loc1 = video_file.find('v_')
        loc2 = video_file.find('_g')
        
        if loc1 != -1 and loc2 != -1:
            action = video_file[(loc1 + 2): loc2]
            actions.append(action)
            all_names.append(video_file)
            print(f"Video: {video_file} -> Action: {action}")
        else:
            print(f"WARNING: Could not parse video filename: {video_file}")

print(f"\nParsed {len(actions)} videos")
print(f"Unique actions: {set(actions)}")

# 检查是否有未知标签
unknown_actions = set(actions) - set(action_names)
if unknown_actions:
    print(f"\nWARNING: Unknown actions found: {unknown_actions}")
else:
    print("\nAll actions are valid!")
