from packaging.version import Version
from metamotivo.fb_cpr.huggingface import FBcprModel
from huggingface_hub import hf_hub_download
from humenv import make_humenv
import gymnasium
from gymnasium.wrappers import FlattenObservation, TransformObservation
from metamotivo.buffers.buffers import DictBuffer
from humenv.env import make_from_name
from humenv import rewards as humenv_rewards

import torch
import math
import h5py
from pathlib import Path
import numpy as np
import cv2, csv, imageio
import mujoco 

def main():
    joint_mapping = {
                "R_Shoulder_z": "right_s0",
                "R_Shoulder_x": "right_s1",
                "R_Shoulder_y": "right_e0",
                "R_Elbow_x": "right_e1",
                "R_Elbow_y": "right_w0",
                "R_Wrist_x": "right_w1",
                "R_Wrist_y": "right_w2",
                "L_Shoulder_z": "left_s0",
                "L_Shoulder_x": "left_s1",
                "L_Shoulder_y": "left_e0",
                "L_Elbow_x": "left_e1",
                "L_Elbow_y": "left_w0",
                "L_Wrist_x": "left_w1",
                "L_Wrist_y": "left_w2"
    }
    model = FBcprModel.from_pretrained("facebook/metamotivo-S-1")
    device = "cpu"
    if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):
        transform_obs_wrapper = lambda env: TransformObservation(
                env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
            )
    else:
        transform_obs_wrapper = lambda env: TransformObservation(
                env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device), env.observation_space
            )

    env, info = make_humenv(
        num_envs=1,
        wrappers=[
            FlattenObservation,
            transform_obs_wrapper,
        ],
        state_init="Default",
    )
    # Print Environment space info
    print("Observation space", (env.observation_space))
    print(f"Action space: {env.action_space}")
    mod = env.unwrapped.model
    data = env.unwrapped.data
    print("Model", mod)
    print("Data", env.unwrapped.data)

    model.to(device)
    z = model.sample_z(1)
    observation, info = env.reset() 

    frames = [env.render()]
    joint_angles = []
    for i in range(300):
        action = model.act(observation, z, mean=True)
        observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
        frames.append(env.render())
        frame = {}
        for i in range(37, mod.njnt):
            if mod.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                qpos_start = mod.jnt_qposadr[i]
                frame[mujoco.mj_id2name(mod, mujoco.mjtObj.mjOBJ_JOINT, i)] = data.qpos[qpos_start]
        joint_angles.append(frame)

    baxter_joints = []
    for joint_frame in joint_angles:
        baxter_frame = {}
        for joint_name, joint_angle in joint_frame.items():
            if joint_mapping.get(joint_name, 0) != 0:
                baxter_frame[joint_mapping[joint_name]] = joint_angle
        baxter_joints.append(baxter_frame)
    # print(baxter_joints)

    with open("gestures.csv", 'w') as gestures:
        # writer = csv.writer(gestures, delimiter=" ")
        
        for row in baxter_joints:
            # print(row)
            cssv = ",".join(f'"{key}":{val}' for key, val in row.items())
            print("{" + cssv + "}")
            gestures.write("{" + cssv + "}\n")
    env.close()
    return frames
if __name__ == "__main__":
    frames = main()
    imageio.mimsave("output.gif", frames, fps=15)  # Adjust fps as needed
    # for frame in frames:
    #     cv2.imshow("Orig Frame", frame)
    #     if cv2.waitKey(0) & 0xFF == ord('q'):
    #         break
