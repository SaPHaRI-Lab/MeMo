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
import cv2
import mujoco 

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
print(f"Action space: {env.action_space}") # Box Type: 
print(f"Meta data: {env.metadata}") # Box Type: 
mod = env.unwrapped.model
data = env.unwrapped.data
print("Model", mod)
print("Data", env.unwrapped.data)

joints = [mod.joint(i) for i in range(37, mod.njnt)]
joint_names = [mod.joint(i).name for i in range(37, mod.njnt)]

print("Joint Names", joint_names)
print("Positions", [joint.pos for joint in joints])
print("Num joints", mod.njnt)

model.to(device)
z = model.sample_z(1)
observation, info = env.reset() 

joints = [mod.joint(i) for i in range(37, mod.njnt)]
joint_names = [mod.joint(i).name for i in range(37, mod.njnt)]

frames = [env.render()]
for i in range(1):
    action = model.act(observation, z, mean=True)
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
    frames.append(env.render())  
    # print("Positions", (info["qpos"][37:]))
    # print("Num Positions", len(info["qpos"][37:]))
    # print("Velocities", (info["qvel"]))

for i in range(37, mod.njnt):
    joint_name = mujoco.mj_id2name(mod, mujoco.mjtObj.mjOBJ_JOINT, i)
    qpos_start = mod.jnt_qposadr[i]
    if mod.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
        joint_angle = data.qpos[qpos_start]
        print(f"Hinge Joint '{joint_name}': {joint_angle:.4f} radians")
    elif mod.jnt_type[i] == mujoco.mjtJoint.mjJNT_BALL:
        joint_quat = data.qpos[qpos_start:qpos_start + 4]
        print(f"Ball Joint '{joint_name}': Quaternion {joint_quat}")
    elif mod.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
        position = data.qpos[qpos_start:qpos_start + 3]
        orientation = data.qpos[qpos_start + 3:qpos_start + 7]
        print(f"Free Joint '{joint_name}': Position {position}, Orientation (Quaternion) {orientation}")

for frame in frames:
    cv2.imshow("Orig Frame", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
env.close()
