from huggingface_hub import hf_hub_download
import h5py
from metamotivo.buffers.buffers import DictBuffer
local_dir = "metamotivo-S-1-datasets"
dataset = "buffer_inference_500000.hdf5"  # a smaller buffer that can be used for reward inference
# dataset = "buffer.hdf5"  # the full training buffer of the model
buffer_path = hf_hub_download(
        repo_id="facebook/metamotivo-S-1",
        filename=f"data/{dataset}",
        repo_type="model",
        local_dir=local_dir,
    )
hf = h5py.File(buffer_path, "r")
print(hf.keys())

# create a DictBuffer object that can be used for sampling
data = {k: v[:] for k, v in hf.items()}
buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
buffer.extend(data)