import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

top_folder = "/home/famadio/Workspace/RL/isaac_lab_playground/IsaacLab/logs/rsl_rl/kangaroo_flat/"
subfolders = [f.path for f in os.scandir(top_folder) if f.is_dir()]
print("Subfolders:", subfolders)

tags = [
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Train/mean_reward",
    "Episode_Termination/time_out",
    "Episode_Termination/falling",
]

data = {tag: [] for tag in tags}

for dir in subfolders:
    # Find the file with extension ".0" inside the directory
    event_file = next((f for f in os.listdir(dir) if f.endswith(".0")), None)
    if event_file:
        event_file_path = os.path.join(dir, event_file)
        print("Event file path:", event_file_path)
    else:
        print(f"No event file found in {dir}")
        continue

    # Initialize EventAccumulator
    event_acc = EventAccumulator(
        event_file_path,
        size_guidance={
            "scalars": 0,  # 0 = load all
        },
    )
    event_acc.Reload()  # Load the events

    # # List available scalar tags
    # all_tags = event_acc.Tags()["scalars"]
    # print("Available scalar tags:", all_tags)

    for tag in tags:
        events = event_acc.Scalars(tag)
        data[tag].append([(e.value) for e in events])

for tag in tags:
    label = tag.replace("/", "_").lower()
    # Save the data to a .npy file
    output_file = Path(top_folder, label)
    np.save(output_file, np.array(data[tag]))
    print(f"Data saved to {output_file}")
