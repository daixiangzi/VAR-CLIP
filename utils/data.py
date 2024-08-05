import os.path as osp
import os
import PIL.Image as PImage
from pathlib import Path
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import Dataset
from tokenizer import tokenize

# For CLIP
clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def image_transform(final_reso: int, model='train',
    hflip=False, mid_reso=1.125,):
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ] 
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    if model == 'train':
       return train_aug
    else:
        return val_aug


def read_path_caption(file_txt):
    image_captions_list = []
    with open(file_txt, "r") as file:
        for line in file:
            line = line.strip()
            if line and ':' in line:  # Skip empty lines and lines without ':'
                parts = line.split(':')
                if len(parts) == 2:
                    a, b = parts
                    image_captions_list.append((a.strip(), b.strip()))
    return image_captions_list


# 读取数据
class ImageNet(Dataset):
    '''
    Args:
        root: file root
        split: data split
    
        Returns:
            pic(PIL.Image.Image)   # return RGB
    '''

    def __init__(
            self,
            root: str,
            final_reso: int, model='train',
    hflip=False, mid_reso=1.125,
    ) -> None:
        super(ImageNet, self).__init__()
        assert model in ["train", "val"]
        self.img_transform = image_transform(final_reso, model,
    hflip, mid_reso)
        self.data_dir = Path(root)  
        self.files = str(self.data_dir / f"{model}" / "image_captions.txt")   # your caption file
        self.reader = read_path_caption(self.files)  

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, indices):
        img_path, captions = self.reader[indices]
        img = pil_loader(img_path)
        img1 = self.img_transform(img)
        caption = tokenize(captions)  
        
        return img1, caption   

  
def imagenet(root, final_reso, model, hflip, mid_reso):
    return ImageNet(root, final_reso, model, hflip, mid_reso)