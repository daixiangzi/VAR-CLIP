{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "### 🚀 For an interactive experience, head over to our [demo platform](https://var.vision/demo) and dive right in! 🌟"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "is_executing": true
   },
   "outputs": [],
   "source": [
    "################## 1. Download checkpoints and build models\n",
    "import os\n",
    "import os.path as osp\n",
    "import torch, torchvision\n",
    "import random\n",
    "import numpy as np\n",
    "import torch.nn.functional as F\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from clip_util import CLIPWrapper\n",
    "from models.clip import clip_vit_l14   \n",
    "from tokenizer import tokenize\n",
    "import PIL.Image as PImage, PIL.ImageDraw as PImageDraw\n",
    "from pathlib import Path\n",
    "setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed\n",
    "setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed\n",
    "from models import VQVAE, build_vae_var\n",
    "\n",
    "MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====\n",
    "assert MODEL_DEPTH in {16, 20, 24, 30}\n",
    "\n",
    "\n",
    "# download checkpoint\n",
    "hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'\n",
    "vae_ckpts, var_ckpts = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'\n",
    "vae_ckpt, var_ckpt = 'pretrained/vae_ch160v4096z32.pth', 'local_output/ar-ckpt-last.pth'\n",
    "if not osp.exists(vae_ckpt): os.system(f'wget -P pretrained {hf_home}/{vae_ckpts}')\n",
    "if not osp.exists(var_ckpt): os.system(f'wget -P pretrained {hf_home}/{var_ckpts}')\n",
    "\n",
    "\n",
    "\n",
    "# build vae, var\n",
    "patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "if 'vae' not in globals() or 'var' not in globals():\n",
    "    vae, var = build_vae_var(\n",
    "        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters\n",
    "        device=device, patch_nums=patch_nums,\n",
    "        n_cond_embed=768, depth=MODEL_DEPTH, shared_aln=False,\n",
    "    )\n",
    "# load clip_vit_l14\n",
    "normalize_clip = True\n",
    "clip = clip_vit_l14(pretrained=True).to(device).eval()\n",
    "clip = CLIPWrapper(clip, normalize=normalize_clip)\n",
    "\n",
    "# load checkpoints\n",
    "vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)\n",
    "var.load_state_dict(torch.load(var_ckpt, map_location='cpu')['trainer']['var_wo_ddp'], strict=True)\n",
    "vae.eval(), var.eval()\n",
    "for p in vae.parameters(): p.requires_grad_(False)\n",
    "for p in var.parameters(): p.requires_grad_(False)\n",
    "print(f'prepare finished.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "############################# 2. Sample with classifier-free guidance\n",
    "import matplotlib.pyplot as plt\n",
    "from torchvision import transforms\n",
    "import time\n",
    "import torch\n",
    "from sklearn.decomposition import PCA\n",
    "import numpy as np\n",
    "import PIL.Image as PImage\n",
    "\n",
    "# set args\n",
    "seed = 0 #@param {type:\"number\"}\n",
    "torch.manual_seed(seed)\n",
    "num_sampling_steps = 250 #@param {type:\"slider\", min:0, max:1000, step:1}\n",
    "cfg = 4 #@param {type:\"slider\", min:1, max:10, step:0.1}\n",
    "more_smooth = False # True for more smooth output\n",
    "\n",
    "text = 'a sailboat with a sail at sunset'\n",
    "tmp_text = [text] * 16\n",
    "bs = len(tmp_text)\n",
    "\n",
    "texts = tokenize(tmp_text).to(device)    # (1, 77)\n",
    "text_embeddings = clip.encode_text(texts)  # [1, 768]\n",
    "embeds = text_embeddings.expand(bs, -1)\n",
    "\n",
    "# seed\n",
    "torch.manual_seed(seed)\n",
    "random.seed(seed)\n",
    "np.random.seed(seed)\n",
    "torch.backends.cudnn.deterministic = True\n",
    "torch.backends.cudnn.benchmark = False\n",
    "\n",
    "# run faster\n",
    "tf32 = True\n",
    "torch.backends.cudnn.allow_tf32 = bool(tf32)\n",
    "torch.backends.cuda.matmul.allow_tf32 = bool(tf32)\n",
    "torch.set_float32_matmul_precision('high' if tf32 else 'highest')\n",
    "\n",
    "\n",
    "# sample\n",
    "B = len(embeds)\n",
    "with torch.inference_mode():\n",
    "    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster\n",
    "        recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=embeds, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)\n",
    "\n",
    "chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)   \n",
    "chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()\n",
    "chw = PImage.fromarray(chw.astype(np.uint8))\n",
    "chw.show()\n",
    "chw.save(\"gen.png\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
