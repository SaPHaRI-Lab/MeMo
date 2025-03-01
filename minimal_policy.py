from humenv import make_humenv
from gymnasium.wrappers import FlattenObservation, TransformObservation
import torch
from metamotivo.fb_cpr.huggingface import FBcprModel

device = "cpu"
env, _ = make_humenv(
    num_envs=1,
    wrappers=[
        FlattenObservation,
        lambda env: TransformObservation(
            env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device), env.observation_space # For gymnasium <1.0.0 remove the last argument: env.observation_space
        ),
    ],
    state_init="Default",
)

model = FBcprModel.from_pretrained("facebook/metamotivo-S-1")
model.to(device)
z = model.sample_z(1)
observation, _ = env.reset()
for i in range(10):
    action = model.act(observation, z, mean=True)
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
    print(info)