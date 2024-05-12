import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = "soccer.jpeg"

img = Image.open(img_path)
img_arr = np.array(img)


img_arr[:, :, 1:] = 0
plt.imshow(img_arr)
plt.waitforbuttonpress()
