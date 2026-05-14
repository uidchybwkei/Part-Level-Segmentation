import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def prepare_image_from_pil(pil_image, short_edge=640):
    t = transforms.Compose([transforms.Resize(short_edge, interpolation=Image.BICUBIC)])
    image_ori = np.asarray(t(pil_image.convert("RGB")))
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
    return image_ori, images
