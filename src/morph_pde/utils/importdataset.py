import numpy as np
import cv2
import os

class ImportImgData:
    @staticmethod
    # import dataset

    def load_A0_images(df,pathA0,imsize):
        imagesA0 = []
        for i in df.index.values:
            #print(i)
            base = os.path.sep.join([pathA0, "{}_CgA0.png".format(i + 1)]) # for S0 images --> "{}_CgS0.png"
            #print(base)
            image = cv2.imread(base,0) # read the path using opencv
            image = cv2.resize(image, (imsize,imsize))
            #print(image.shape)
            #plt.imshow(image) # use matplotlib to plot the image
            image = image[:,:,np.newaxis] #This is convert (600,600) --> (600,600,1)
            imagesA0.append(image) 
        return np.array(imagesA0)

    def load_S0_images(df,pathS0,imsize):
        imagesS0 = []
        for j in df.index.values:
            base2 = os.path.sep.join([pathS0, "{}_CgS0.png".format(j + 1)]) # for S0 images --> "{}_CgS0.png"
            #print(base)
            image2 = cv2.imread(base2,0) # read the path using opencv
            image2 = cv2.resize(image2, (imsize,imsize))
            #print(image2.shape)
            #plt.imshow(image2) # use matplotlib to plot the image
            image2 = image2[:,:,np.newaxis] #This is convert (600,600) --> (600,600,1)
            imagesS0.append(image2) 
        return np.array(imagesS0)