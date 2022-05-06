import torch

from merlinsar.test.model import *
from merlinsar.test.utils import *
from merlinsar.test.model_test import *
import os
from glob import glob

import numpy as np
from merlinsar.test.load_cosar import cos2mat

M = 10.089038980848645
m = -1.429329123112601

this_dir, this_filename = os.path.split(__file__)


def despeckle(image_path, destination_directory, stride_size=64,
              model_weights_path=os.path.join(this_dir, "saved_model", "model.pth"), patch_size=256, height=256,
              width=256):
    """ The default despeckling function. If you own a powerful GPU and you
    wish to despeckle a high-res image, feel free to adapt the height and width arguments provided by default.
    Results are saved in the directory provided in the 'destination directory'

            Parameters
            ----------
            image_path: string
            the path leading to the image to be despeckled i.e denoised from speckle

            destination_directory: string
            path leading to the results folder

            stride_size: integer or tuple
            stride of the autoencoder

            model_weights_path: string
            path leading to the weights of our pre-trained model. Value by default is our weights.

            patch_size: integer
            Area size of the sub-image to be processed. Value by default is 256.

            height: integer
            Height of the image. Value by default is 256.

            width: integer
            Width of the image. Value by default is 256.

    """

    denoiser = Denoiser()

    if not os.path.exists(destination_directory + '/processed_image'):
        os.mkdir(destination_directory + '/processed_image')

    test_data = destination_directory + '/processed_image'
    image_data = cos2mat(image_path)

    np.save(test_data + '/test_image_data.npy', image_data)

    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), destination_directory, destination_directory))

    test_files = glob((test_data + '/*.npy'))
    print(test_files)

    denoiser.test(test_files, model_weights_path, save_dir=destination_directory,
                  stride=stride_size, patch_size=patch_size, height=height, width=width)


def despeckle_from_coordinates(image_path, coordinates_dict, destination_directory, stride_size=64,
                               model_weights_path=os.path.join(this_dir, "saved_model", "model.pth"), patch_size=256,
                               height=256, width=256):
    """ The despeckling function with a coordinates argument. The ideal choice if you need to despeckle only a
    certain area of your high-res image. Results are saved in the directory provided in the 'destination directory'

            Parameters
            ----------
            image_path: string
            the path leading to the image to be despceckled

            coordinates_dict: a list of coordinates the list of coordinates identifying the sub-image you wish to
            despeckle e.g: {'x_start':coord1, 'y_start':coord2, 'x_end': coord3, 'y_end':coord4}

            destination_directory: string
            path leading to the results folder

            stride_size: integer or tuple
            stride of the autoencoder

            model_weights_path: string
            path leading to the weights of our pre-trained model. Value by default is our weights.

            patch_size: integer
            Area size of the sub-image to be processed. Value by default is 256.

            height: integer
            Height of the image. Value by default is 256.

            width: integer
            Width of the image. Value by default is 256.

            Returns
            ----------
            None

    """

    x_start = coordinates_dict["x_start"]
    x_end = coordinates_dict["x_end"]
    y_start = coordinates_dict["y_start"]
    y_end = coordinates_dict["y_end"]

    denoiser = Denoiser()

    if not os.path.exists(destination_directory + '/processed_image'):
        os.mkdir(destination_directory + '/processed_image')

    test_data = destination_directory + '/processed_image'

    filelist = glob(os.path.join(test_data, "*"))
    for f in filelist:
        os.remove(f)

    image_data = cos2mat(image_path)

    np.save(test_data + '/test_image_data.npy', image_data[x_start:x_end, y_start:y_end, :])

    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), destination_directory, destination_directory))

    test_files = glob((test_data + '/*.npy'))
    print(test_files)
    denoiser.test(test_files, model_weights_path, save_dir=destination_directory,
                  stride=stride_size, patch_size=patch_size, height=height, width=width)


def despeckle_from_crop(image_path, destination_directory, stride_size=64,
                        model_weights_path=os.path.join(this_dir, "saved_model", "model.pth"), patch_size=256,
                        height=256,
                        width=256, fixed=True):
    """ The despeckling function with an integrated cropping tool made with OpenCV.
    The ideal choice if you need to despeckle only a certain area of your high-res image. Results are saved in the
    directory provided in the 'destination directory'

            Parameters
            ----------
            image_path: string
            the path leading to the image to be despceckled

            destination_directory: string
            path leading to the results folder

            stride_size: integer or tuple
            stride of the autoencoder

            model_weights_path: string
            path leading to the weights of our pre-trained model. Value by default is our weights.

            patch_size: integer
            Area size of the sub-image to be processed. Value by default is 256.

            height: integer
            Height of the image. Value by default is 256.

            width: integer
            Width of the image. Value by default is 256.

            fixed: bool
            If True, crop size is limited to 256*256

            Returns
            ----------
            None

    """

    denoiser = Denoiser()

    if not os.path.exists(destination_directory + '\\processed_image'):
        os.mkdir(destination_directory + '\\processed_image')

    test_data = destination_directory + '\\processed_image'

    # FROM IMAGE PATH RETRIEVE PNG, NPY, REAL , IMAG, THRESHOLD, FILENAME
    image_png, image_data, image_data_real, image_data_imag, threshold, filename = get_info_image(image_path,
                                                                                                  destination_directory)

    # CROPPING OUR PNG AND REFLECT THE CROP ON REAL AND IMAG
    cropping = False
    if fixed:
        crop_fixed(image_png, image_data_real, image_data_imag, destination_directory, test_data, cropping)
    else:
        crop(image_png, image_data_real, image_data_imag, destination_directory, test_data, cropping)

    image_data_real_cropped = np.load(test_data + '\\image_data_real_cropped.npy')
    store_data_and_plot(image_data_real_cropped, threshold, test_data + '\\image_data_real_cropped.npy')
    image_data_imag_cropped = np.load(test_data + '\\image_data_imag_cropped.npy')
    store_data_and_plot(image_data_imag_cropped, threshold, test_data + '\\image_data_imag_cropped.npy')

    image_data_real_cropped = image_data_real_cropped.reshape(image_data_real_cropped.shape[0],
                                                              image_data_real_cropped.shape[1], 1)
    image_data_imag_cropped = image_data_imag_cropped.reshape(image_data_imag_cropped.shape[0],
                                                              image_data_imag_cropped.shape[1], 1)

    np.save(test_data + '/test_image_data_cropped.npy',
            np.concatenate((image_data_real_cropped, image_data_imag_cropped), axis=2))

    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), destination_directory, destination_directory))

    test_files = glob((test_data + '/test_image_data_cropped.npy'))
    print(test_files)
    denoiser.test(test_files, model_weights_path, save_dir=destination_directory,
                  stride=stride_size, patch_size=patch_size, height=height, width=width)
