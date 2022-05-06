import gc
import os
import sys

import numpy as np
from PIL import Image
from scipy import special
from scipy import signal
from glob import glob
from merlinsar.train.GenerateDataset import GenerateDataset

basedir = '.'

# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle


def symetrisation_patch(ima):
    S = np.fft.fftshift(np.fft.fft2(ima[:, :, 0] + 1j * ima[:, :, 1]))
    p = np.zeros((S.shape[0]))  # azimut (ncol)
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i, :]))
    sp = p[::-1]
    c = np.real(np.fft.ifft(np.fft.fft(p) * np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(), p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1 - 1) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_1 = np.roll(p, shift_az_1)
    shift_az_2 = int(round(-(d1 - 1 - p.shape[0]) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_2 = np.roll(p, shift_az_2)
    window = signal.gaussian(p.shape[0], std=0.2 * p.shape[0])
    test_1 = np.sum(window * p2_1)
    test_2 = np.sum(window * p2_2)
    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1 >= test_2:
        p2 = p2_1
        shift_az = shift_az_1 / p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2 / p.shape[0]
    S2 = np.roll(S, int(shift_az * p.shape[0]), axis=0)

    q = np.zeros((S.shape[1]))  # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:, j]))
    sq = q[::-1]
    # correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q) * np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(), q.shape[0])
    d2 = d2[0]
    shift_range_1 = int(round(-(d2 - 1) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_1 = np.roll(q, shift_range_1)
    shift_range_2 = int(round(-(d2 - 1 - q.shape[0]) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_2 = np.roll(q, shift_range_2)
    window_r = signal.gaussian(q.shape[0], std=0.2 * q.shape[0])
    test_1 = np.sum(window_r * q2_1)
    test_2 = np.sum(window_r * q2_2)
    if test_1 >= test_2:
        q2 = q2_1
        shift_range = shift_range_1 / q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2 / q.shape[0]

    Sf = np.roll(S2, int(shift_range * q.shape[0]), axis=1)
    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))
    return np.stack((np.real(ima2), np.imag(ima2)), axis=2)


def symetrisation_patch_test(real_part, imag_part):
    S = np.fft.fftshift(np.fft.fft2(real_part[0, :, :, 0] + 1j * imag_part[0, :, :, 0]))
    p = np.zeros((S.shape[0]))  # azimut (ncol)
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i, :]))
    sp = p[::-1]
    c = np.real(np.fft.ifft(np.fft.fft(p) * np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(), p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1 - 1) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_1 = np.roll(p, shift_az_1)
    shift_az_2 = int(round(-(d1 - 1 - p.shape[0]) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_2 = np.roll(p, shift_az_2)
    window = signal.gaussian(p.shape[0], std=0.2 * p.shape[0])
    test_1 = np.sum(window * p2_1)
    test_2 = np.sum(window * p2_2)
    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1 >= test_2:
        p2 = p2_1
        shift_az = shift_az_1 / p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2 / p.shape[0]
    S2 = np.roll(S, int(shift_az * p.shape[0]), axis=0)

    q = np.zeros((S.shape[1]))  # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:, j]))
    sq = q[::-1]
    # correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q) * np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(), q.shape[0])
    d2 = d2[0]
    shift_range_1 = int(round(-(d2 - 1) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_1 = np.roll(q, shift_range_1)
    shift_range_2 = int(round(-(d2 - 1 - q.shape[0]) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_2 = np.roll(q, shift_range_2)
    window_r = signal.gaussian(q.shape[0], std=0.2 * q.shape[0])
    test_1 = np.sum(window_r * q2_1)
    test_2 = np.sum(window_r * q2_2)
    if test_1 >= test_2:
        q2 = q2_1
        shift_range = shift_range_1 / q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2 / q.shape[0]

    Sf = np.roll(S2, int(shift_range * q.shape[0]), axis=1)
    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))
    ima2 = ima2.reshape(1, np.size(ima2, 0), np.size(ima2, 1), 1)
    return np.real(ima2), np.imag(ima2)


class train_data():
    """ A class for loading and shuffling our training data
    """

    def __init__(self, filepath="%s/data/image_clean_pat.npy" % basedir):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath="%s/data/image_clean_pat.npy" % basedir):
    """ A function that loads data using the class train_data

            Parameters
            ----------
            filepath : string
            the path leading

            Returns
            -------
            train_data: nd.array

    """
    return train_data(filepath=filepath)


def normalize_sar(im):
    """ Normalization of a numpy-stored image

            Parameters
            ----------
            im : an image object

            Returns
            ----------
            the normalised im

        """
    return ((np.log(im + 1e-6) - m) / (M - m)).astype(np.float32)


def denormalize_sar(im):
    """ De-normalization of a numpy-stored image

            Parameters
            ----------
            im : an image object

            Returns
            ----------
            the denormalised im

        """
    return np.exp((np.clip(np.squeeze(im), 0, 1)) * (M - m) + m)


def load_train_data(filepath, patch_size, batch_size, stride_size,
                    n_data_augmentation):  # TODO: add control on training data: exit if does not exists
    datagen = GenerateDataset()
    imgs = datagen.generate_patches(src_dir=filepath, pat_size=patch_size, step=0,
                                    stride=stride_size, bat_size=batch_size, data_aug_times=n_data_augmentation)
    return imgs


def load_sar_images(filelist):
    """ Loads files , resize them and append them into a list called data

            Parameters
            ----------
            filelist : a path to a folder containing the images

            Returns
            ----------
            A list of images

    """
    if not isinstance(filelist, list):
        im = np.load(filelist)
        return np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 2)
    data = []
    for file in filelist:
        im = np.load(file)
        data.append(np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 2))
    return data


def store_data_and_plot(im, threshold, filename):
    """ Creates an image memory from an object exporting the array interface and returns a converted copy of this
        image into greyscale mode ("L")

            Parameters
            ---------
            im : the image to store
            threshold: clip a maximum value in the image array i.e. values are to be between 0 and threshold
            filename: the path to store the result array image in .png

            Returns
            ----------
            filename : string
            the path leading to the saved object

    """
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy', 'png'))


def save_sar_images(denoised, noisy, imagename, save_dir, groundtruth=None):
    """ Saves both denoised and noisy images in the save directory provided

            Parameters
            ----------
            denoised: nd.array
            the denoised image stored in an array

            noisy: nd.array
            the noisy image stored in an array

            imagename: string
            the name given to the image

            save_dir: string
            the path for saving results

            groundtruth: bool
            Special case if the user has groundtruth images. Results will be saved with 'groundtruth' in their names

            Returns
            ----------
            None

    """
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'domancy': 560, 'Brazil': 103.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(noisy) + 3 * np.std(noisy)

    if groundtruth:
        groundtruthfilename = save_dir + "/groundtruth_" + imagename
        groundtruthfilename = groundtruthfilename.replace("\\", "")

        np.save(groundtruthfilename, groundtruth)
        store_data_and_plot(groundtruth, threshold, groundtruthfilename)

    denoisedfilename = save_dir + "/denoised_" + imagename
    denoisedfilename = denoisedfilename.replace("\\", "")

    np.save(denoisedfilename, denoised)
    store_data_and_plot(denoised, threshold, denoisedfilename)

    noisyfilename = save_dir + "/noisy_" + imagename
    noisyfilename = noisyfilename.replace("\\", "")

    np.save(noisyfilename, noisy)
    store_data_and_plot(noisy, threshold, noisyfilename)


def save_real_imag_images(real_part, imag_part, imagename, save_dir):
    """ Saves denoised real and imaginary parts of images in the save directory provided

            Parameters
            ----------
            real_part: nd.array
            the real part of the image stored in an array

            imag_part: nd.array
            the imaginary part of the image stored in an array

            imagename: string
            the name given to the image

            save_dir: string
            the path for saving results

            Returns
            ----------
            None

    """
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'Brazil': 103.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(imag_part) + 3 * np.std(imag_part)

    realfilename = save_dir + "/denoised_real_" + imagename
    realfilename = realfilename.replace("\\", "")

    np.save(realfilename, real_part)
    store_data_and_plot(real_part, threshold, realfilename)

    imagfilename = save_dir + "/denoised_imag_" + imagename
    imagfilename = imagfilename.replace("\\", "")

    np.save(imagfilename, imag_part)
    store_data_and_plot(imag_part, threshold, imagfilename)


def save_real_imag_images_noisy(real_part, imag_part, imagename, save_dir):
    """ Saves noisy real and imaginary parts of images in the save directory provided

            Parameters
            ----------
            real_part: nd.array
            the real part of the image stored in an array

            imag_part: nd.array
            the imaginary part of the image stored in an array

            imagename: string
            the name given to the image

            save_dir: string
            the path for saving results

            Returns
            ----------
            None

        """
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'Brazil': 103.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(np.abs(imag_part)) + 3 * np.std(np.abs(imag_part))

    realfilename = save_dir + "/noisy_real_" + imagename
    realfilename = realfilename.replace("\\", "")

    np.save(realfilename, real_part)
    store_data_and_plot(np.sqrt(2) * np.abs(real_part), threshold, realfilename)

    imagfilename = save_dir + "/noisy_imag_" + imagename
    imagfilename = imagfilename.replace("\\", "")

    np.save(imagfilename, imag_part)
    store_data_and_plot(np.sqrt(2) * np.abs(imag_part), threshold, imagfilename)


def cal_psnr(Shat, S):
    # takes amplitudes in input
    # Shat: a SAR amplitude image
    # S:    a reference SAR image
    P = np.quantile(S, 0.99)
    res = 10 * np.log10((P ** 2) / np.mean(np.abs(Shat - S) ** 2))
    return res
