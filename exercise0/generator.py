# %%%
import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.index = 0
        self.epoch = 0

        image_list = os.listdir(file_path)
        label_dict = json.load(open(label_path))

        # self.img_names = np.array(list(label_dict.keys()))
        # self.labels = np.array(list(label_dict.values()))

        self.imageDB = []
        self.labelDB = []
        self.imageName = []

        for image in image_list:
            if image_size:
                self.imageDB.append(skimage.transform.resize(np.load(file_path + image), image_size))
            else:    
                self.imageDB.append(np.load(file_path + image))

            self.labelDB.append(label_dict[image[ : -4]])
            self.imageName.append(image[ : -4])

        # ImageGenerator('./exercise_data/', './Labels.json', 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        # plt.imshow(self.imageDB[1])

        return

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        batch_images = []
        batch_labels = []

        self.index += self.batch_size

        # base case
        if self.index <= (self.epoch + 1) * (len(self.imageDB) - 1):
            batch_images[0 : self.batch_size] = self.imageDB[self.index - self.batch_size : self.index]
            batch_labels[0 : self.batch_size] = self.labelDB[self.index - self.batch_size : self.index]

            # if data_size % batch_size == 0
            if self.index == (self.epoch + 1) * (len(self.imageDB) - 1):
                self.epoch += 1

        # if data_size % batch_size != 0
        else:
            overflow_pont = len(self.imageDB) - (self.index - self.batch_size)

            # fill batch first with remaining unused images
            batch_images[0 : overflow_pont] = self.imageDB[self.index - self.batch_size : len(self.imageDB)]
            batch_labels[0 : overflow_pont] = self.labelDB[self.index - self.batch_size : len(self.labelDB)]

            # fill remaining portion of batch with cycling data
            self.epoch += 1

            batch_images[overflow_pont : self.batch_size] = self.imageDB[0 : self.batch_size - overflow_pont]
            batch_labels[overflow_pont : self.batch_size] = self.labelDB[0 : self.batch_size - overflow_pont]

        return batch_images, batch_labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict(x)

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        return
# %%
if __name__ == "__main__":
    gen = ImageGenerator('./exercise_data/', './Labels.json', 10, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
    images, labels = gen.next()
    print(len(labels))
    print(labels[1])
    images, labels = gen.next()
    images, labels = gen.next()
    images, labels = gen.next()
    images, labels = gen.next()
    images, labels = gen.next()
    images, labels = gen.next()
    images, labels = gen.next()
    images, labels = gen.next()
    print(len(labels))
    print(labels[6])
    images, labels = gen.next()
    print(len(labels))
    print(labels[1])
    
    print(gen.class_name(5))
    

# %%
