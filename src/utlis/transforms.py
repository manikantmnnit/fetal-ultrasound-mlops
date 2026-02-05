import numpy as np
import cv2 as cv
from dataclasses import dataclass
from PIL import Image

@dataclass
class CroppedImage:

    """Crops the given PIL Image by removing specified pixels from the top and left sides."""
    top: int = 0
    left: int = 0


    def __call__(self, img):

        assert len(img.size)==2, "Input image must be a 2D tensor"

         # PIL crop expects (left, top, right, bottom)
        return img.crop((self.left, self.top,  img.width-self.left, img.height-self.top ))

class ApplyCLAHE:
  def __init__(self,clip_limit=2.0,tile_grid_size=(8,8)):
    self.clahe=cv.createCLAHE(clipLimit=clip_limit,tileGridSize=tile_grid_size)

  def __call__(self,img):

    # img=img.convert('L')
    img_np=np.array(img)
    img=self.clahe.apply(img_np)
    img=Image.fromarray(img)
    return img
