import cv2
import pydicom as dicom
import skimage.exposure as hist
import numpy as np
import os


def generate_jpg(path, new_path, downsample_size=224, adaptive_norm=True):

    image_array = dicom.read_file(path).pixel_array

    downsampled = zoom_2D(image_array, (downsample_size, downsample_size))

    if adaptive_norm:
        downsampled = adaptive_normalization_param(downsampled)

    downsampled = np.array(downsampled, dtype=np.float32)
    downsampled = cv2.normalize(downsampled, downsampled, 0, 255, cv2.NORM_MINMAX)
    downsampled = np.array(downsampled, dtype=np.uint8)

    if not new_path.endswith(".jpg"):
        return "ERROR"

    cv2.imwrite(new_path, downsampled, [cv2.IMWRITE_JPEG_QUALITY, 100])


def zoom_2D(image, new_shape):
    """
    Uses open CV to resize a 2D image
    :param image: The input image, numpy array
    :param new_shape: New shape, tuple or array
    :return: the resized image
    """

    # OpenCV reverses X and Y axes
    return cv2.resize(
        image, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_CUBIC
    )


def adaptive_normalization_param(tensor, dim3d=False):
    """
    Contrast localized adaptive histogram normalization
    :param tensor: ndarray, 2'd or 3d
    :param dim3d: 2D or 3d. If 2d use Scikit, if 3D use the MCLAHe implementation
    :return: normalized image
    """
    return hist.equalize_adapthist(tensor, kernel_size=128, nbins=1024)
