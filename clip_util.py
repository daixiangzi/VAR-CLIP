import math
import os
import shutil
from pathlib import Path
import torch
import torch.distributed as dist


class CLIPWrapper():

    def __init__(self, clip, normalize=True):
        self.clip = clip.eval()
        self.normalize = normalize
        if normalize:
            print("normalize CLIP embeddings")

    @torch.no_grad()
    def encode_image(self, image):
        embeds = self.clip.encode_image(image)
        if self.normalize:
            embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds

    @torch.no_grad()
    def encode_text(self, text):
        embeds = self.clip.encode_text(text)
        if self.normalize:
            embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds