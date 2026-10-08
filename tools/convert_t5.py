# Convert Wan2.2 UMT5-XXL encoder .pth -> ckpts/t5_enc.safetensors (bf16).
#
# Linear weights are saved TRANSPOSED to [in, out] (= torch W.t().contiguous(),
# matmul-ready for sd::matmul). Embedding / norm / pos-bias weights keep their
# original layout. Everything stays bf16 — the C++ side keeps bf16 mmap views
# and lets sd::matmul convert per call.
#
# RAM note: torch.load peak ~11.5 GB; transposes replace dict entries one at a
# time (old freed), so peak stays ~11.5 GB + one 336 MB tensor.
import torch
from safetensors.torch import save_file

SRC = "ckpts/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
DST = "ckpts/t5_enc.safetensors"

sd = torch.load(SRC, map_location="cpu", weights_only=True)
print(f"{len(sd)} tensors")

for k, v in sd.items():
    if v.ndim == 2 and ("embedding.weight" not in k):
        sd[k] = v.t().contiguous()  # [out,in] -> [in,out]; frees old
    assert sd[k].dtype == torch.bfloat16, (k, sd[k].dtype)

save_file(sd, DST)
print("wrote", DST)

import os
print("bytes:", os.path.getsize(DST))
