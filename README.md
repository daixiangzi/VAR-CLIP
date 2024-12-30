
<div align="center">

#  *VAR-CLIP*:<br> Text-to-Image Generator with Visual Auto-Regressive Modeling
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2408.01181-b31b1b.svg)](https://arxiv.org/abs/2408.01181)&nbsp;
</div>

<p align="center">
<img src="img/main.png" width=95%>
<p>

> [**VAR-CLIP: Text-to-Image Generator with Visual Auto-Regressive Modeling**](https://arxiv.org/abs/2408.01181)<br>
> Qian Zhang, [Xiangzi Dai](https://github.com/daixiangzi), Ninghua Yang, [Xiang An](https://github.com/anxiangsir), Ziyong Feng, [Xingyu Ren](https://xingyuren.github.io/)
> <br>Institute of Applied Physics and Computational Mathematics, DeepGlint,Shanghai Jiao Tong University
> 
## Some example for text-conditional generation:
<img src="img/show_res.png" width="800px"/> . 

## Some example for class-conditional generation:
<img src="img/concatenated_image.jpg" width="800px"/> .

### TODO
- [x] Relased Pre_trained model on ImageNet.
- [x] Relased train code.
- [x] Relased Arxiv.
- [x] Training T2I on the ImageNet dataset has been completed.
- [x] Training on the ImageNet dataset has been completed.
  
## Getting Started
### Requirements
```bash
pip install -r requirements.txt
```
### Download Pretrain model/Dataset
<span style="font-siz15px;"> 1. Place the downloaded ImageNet train/val parts separately under **train/val** in the directory **./imagenet/**
</span>   
2. Download **clip/vae** pretrain model put on **pretrained/**
   
>[**Download ClIP_L14**](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)<br>
>[**Download VAE**](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth)<br>
>[**Download VAR_CLIP Model Weight**](https://drive.google.com/file/d/1HlFgY3LysL0yDGSRvpi7bw7TwAztA2Ob/view?usp=drive_link)<br>



## Training Scripts
```bash
# training VAR-CLIP-d16 for 1000 epochs on ImageNet 256x256 costs 4.1 days on 64 A100s
# Before running, you need to configure the IP addresses of multiple machines in the run.py file and data_path
python run.py
```
## demo Scripts
```bash
# you can run demo_samle.ipynb get text-conditional generation resulets after train completed.
demo_sample.ipynb
```
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citations

```bibtex
@misc{zhang2024varclip,
      title={VAR-CLIP: Text-to-Image Generator with Visual Auto-Regressive Modeling}, 
      author={Qian Zhang and Xiangzi Dai and Ninghua Yang and Xiang An and Ziyong Feng and Xingyu Ren},
      year={2024},
      journal={arXiv:2408.01181},
}
```
* VAR - https://github.com/FoundationVision/VAR
* CLIP - https://github.com/openai/CLIP
