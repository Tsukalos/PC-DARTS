import random
import numbers
import collections
from math import floor
import functional as F

Sequence = collections.abc.Sequence
Iterable = collections.abc.Iterable

class ToTensor(object):
    """Convert a ``Numpy Image`` or ``numpy.ndarray`` to tensor.
    Converts a Numpy Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the Numpy Image has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (Numpy Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomCrop(object):
    """Crop the given set of images at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (Numpy Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        if len(img.shape) > 3:
            _, h, w, _ = img.shape
        else:
            h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (Numpy Image): Set of images to be cropped.
        Returns:
            Numpy Image: Set of cropped images.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.shape[1], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.shape[0]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class CenterCrop(object):
    """Crop the given set of images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a center crop.
        Args:
            img (Numpy Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for center crop.
        """
        if len(img.shape) > 3:
            _, h, w, _ = img.shape
        else:
            h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = floor((h - th)/2)
        j = floor((w - tw)/2)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (Numpy Image): Set of images to be cropped.
        Returns:
            Numpy Image: Set of cropped images.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.shape[1], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.shape[0]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class RandomHorizontalFlip(object):
    """Horizontally flip the given set of images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Numpy Image): Set of images to be flipped.
        Returns:
            Numpy array: Randomly flipped set of images.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

class MakeGrid(object):
    """Make a grid of images.
    Converts a 4D Tensor of shape (64 x C x H x W) or a list of 64 images
    all of the same size to a 3D Tensor of shape (C x 8H x 8W).
    Args:
        grid_mode: Type of grid. Should be: image or pixel. Default is image.
             - image: arrange the 64 images into a 8 x 8 grid of H x W pixels
             - pixel: arrange the 64 images into a H x W grid of 8 x 8 pixels
    """

    def __init__(self, grid_mode='image'):
        self.grid_mode = grid_mode

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor or list): 4D Tensor of shape (64 x C x H x W)
                or a list of 64 images all of the same size.
        Returns:
            Tensor: Grid Tensor image.
        """
        return F.make_grid(tensor, self.grid_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(grid_mode={0})'.format(self.grid_mode)

class Concat(object):
    """Join a sequence of images along an existing axis.
    Converts a 4D Tensor of shape (64 x C x H x W) or a list of 64 images
    all of the same size to a 3D Tensor of shape (64C x H x W).
    """

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor or list): 4D Tensor of shape (64 x C x H x W)
                or a list of 64 images all of the same size.
        Returns:
            Tensor: Grid Tensor image.
        """
        return F.concat(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'

