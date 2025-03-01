from packaging.version import Version
from metamotivo.fb_cpr.huggingface import FBcprModel
from huggingface_hub import hf_hub_download
from humenv import make_humenv
import gymnasium
from gymnasium.wrappers import FlattenObservation, TransformObservation
from metamotivo.buffers.buffers import DictBuffer
from humenv.env import make_from_name
from humenv import rewards as humenv_rewards

# import mediapy as media
import torch
import math
import h5py
from pathlib import Path
import numpy as np
import cv2

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

env, _ = make_humenv(
    num_envs=1,
    wrappers=[
        FlattenObservation,
        transform_obs_wrapper,
    ],
    state_init="Default",
)

model.to(device)
z = model.sample_z(1)
# z = torch.rand(1, 256).to(device)
print(z)
print(f"embedding size {z.shape}")
print(f"z norm: {torch.norm(z)}")
print(f"z norm / sqrt(d): {torch.norm(z) / math.sqrt(z.shape[-1])}")
observation, _ = env.reset()
frames = [env.render()]
for i in range(300):
    action = model.act(observation, z, mean=True)
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
    frames.append(env.render())
for frame in frames:
    cv2.imshow("Orig Frame", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
env.close()
