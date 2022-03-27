import numpy as np
from PIL import Image
import os

oth = "../mal/"
for file in os.listdir():
    if file[:4] == "imnf":
        epi = np.array(Image.open(file))
        mal = np.array(Image.open(oth + file))
        epi[(epi == [0, 255, 0]).all(-1)] = mal[(epi == [0, 255, 0]).all(-1)]
        Image.fromarray(epi).save(file)