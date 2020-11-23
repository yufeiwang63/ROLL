import numpy as np
# from scripts.test_multiworld import show_obs
import copy
import cv2
from multiworld.core.image_env import normalize_image

data = np.load('data/local/pre-train-vae/SawyerPushHurdle-v0-no-seg-2000.npy', allow_pickle=True)
data = np.load('data/local/pre-train-lstm/vae-only-SawyerPushHurdle-v0-seg-unet-2000-0.3-0.5-puck-pos.npy', allow_pickle=True)

data = data.item()
print(data.keys())
print(type(data))
for key in data:
    print(key, data[key].shape, data[key].dtype, np.max(data[key]))

def show_obs(normalized_img_vec_, imsize=48, name='img'):
    print(name)
    normalized_img_vec = copy.deepcopy(normalized_img_vec_)
    img = (normalized_img_vec * 255).astype(np.uint8)
    img = img.reshape(3, imsize, imsize).transpose()
    img = img[::-1, :, ::-1]
    cv2.imshow(name, img)
    cv2.waitKey()

for image in data:
    show_obs(image)