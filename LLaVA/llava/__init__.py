# cuDNN 9.20 (pinned by torch cu130) reports SUBLIBRARY_VERSION_MISMATCH on
# GB10/Blackwell; conv falls back to the native CUDA kernel when disabled.
import torch as _torch
_torch.backends.cudnn.enabled = False

from .model import LlavaLlamaForCausalLM
