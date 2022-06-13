import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T


def make_img_to_tensor(img):
    pil_img = np.array(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    img_tensor = T.ToTensor()(pil_img)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor
