# Chase Brown
# SID 106015389
# DeepLearning PA 3: CNN

# data_process: Contains module class for loading data from zip files.
# Load function supplied by Professor Ashis Biswas
import numpy as np
from zipfile import ZipFile


class LoadDataModule(object):
    def __init__(self):
        self.DIR = './'
        pass

    def load(self, mode):
        """
        Load the zip files contents into nparrays
        :param mode: First string of file parse 'train' or 'test'
        :return: array of number of labesl x (pixels for 28x28 images:'784'), array of number of samples x labels
        """
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = self.DIR + label_filename + '.zip'
        image_zip = self.DIR + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

