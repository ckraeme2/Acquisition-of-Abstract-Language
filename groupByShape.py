from PIL import Image
import numpy as np
import os

def buildDatasetAsNpArray(pathToDir):
    data = np.array([], dtype=np.int64).reshape(0,49152)
    for fileName in os.listdir(pathToDir):
        name = os.path.join(pathToDir, fileName)
        image = Image.open(name)
        imageArr = np.asarray(image)
        imageArrFlat = np.reshape(imageArr, (-1, 49152))
        data = np.vstack([data, imageArrFlat])

    return data


data = buildDatasetAsNpArray('./images/content/images')
print(data.shape)