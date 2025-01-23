import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from clip_util import CLIPWrapper
from models.clip import clip_vit_l14   
from tokenizer import tokenize
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from pathlib import Path
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var

MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpts, var_ckpts = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
vae_ckpt, var_ckpt = 'pretrained/vae_ch160v4096z32.pth', 'local_output/ar-ckpt-last.pth'
if not osp.exists(vae_ckpt): os.system(f'wget -P pretrained {hf_home}/{vae_ckpts}')
if not osp.exists(var_ckpt): os.system(f'wget -P pretrained {hf_home}/{var_ckpts}')



# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        n_cond_embed=768, depth=MODEL_DEPTH, shared_aln=False,
    )
# load clip_vit_l14
normalize_clip = True
clip = clip_vit_l14(pretrained=True).to(device).eval()
clip = CLIPWrapper(clip, normalize=normalize_clip)

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu')['trainer']['var_wo_ddp'], strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')

############################# 2. Sample with classifier-free guidance
import matplotlib.pyplot as plt
from torchvision import transforms
import time
import torch
from sklearn.decomposition import PCA
import numpy as np
import PIL.Image as PImage

# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
more_smooth = False # True for more smooth output

text = 'a sailboat with a sail at sunset'
tmp_text = [text] * 16
bs = len(tmp_text)

texts = tokenize(tmp_text).to(device)    # (1, 77)
text_embeddings = clip.encode_text(texts)  # [1, 768]
embeds = text_embeddings.expand(bs, -1)

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')


# sample
B = len(embeds)
with torch.inference_mode():
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
        recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=embeds, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)

chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)   
chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
chw = PImage.fromarray(chw.astype(np.uint8))
chw.show()
chw.save("gen.png")
