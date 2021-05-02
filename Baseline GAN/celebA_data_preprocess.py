import os
import matplotlib.pyplot as plt
import numpy as np
#from scipy.misc import imresize
from PIL import Image

# root path depends on your computer
root = 'data/celebA/celebA/celebA/'
save_root = 'data/resized_celebA/celebA/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = np.array(Image.fromarray(img).resize((resize_size, resize_size), resample=Image.BICUBIC))
    plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

    if (i % 1000) == 0:
        print('%d images complete' % i)