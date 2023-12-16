import torch
import numpy as np
import numbers
import collections

Sequence = collections.abc.Sequence
Iterable = collections.abc.Iterable

def _is_tensor_image(img):
    return torch.is_tensor(img) and (img.ndimension() == 3 or img.ndimension() == 4)

def _is_numpy(img):
    return isinstance(img, np.ndarray)

def _is_numpy_image(img):
    return img.ndim in {2, 3, 4}

def to_tensor(pic):
    """Convert a ``Numpy Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (Numpy Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if not(_is_numpy(pic)):
        raise TypeError('pic should be Numpy Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        if pic.ndim == 3:
            img = torch.from_numpy(np.ascontiguousarray(pic.transpose((2, 0, 1))))
        else:
            img = torch.from_numpy(np.ascontiguousarray(pic.transpose((0, 3, 1, 2))))

        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img.float()

def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if tensor.ndimension() == 3:
        tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    else:
        tensor.sub_(mean[:, :, None, None]).div_(std[:, :, None, None])
    return tensor

def pad(img, padding, fill=0, padding_mode='constant'):
    r"""Pad the given set of images on all sides with specified padding mode and fill value.
    Args:
        img (Numpy Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
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
    Returns:
        Numpy Image: Padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    # DCT image
    if len(img.shape) == 4:
        img = np.pad(img, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
    # RGB image
    if len(img.shape) == 3:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
    # Grayscale image
    if len(img.shape) == 2:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

    return img

def crop(img, i, j, h, w):
    """Crop the given Numpy Image.
    Args:
        img (Numpy Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
    Returns:
        Numpy Image: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(img)))

    return img[..., i:i+h, j:j+w, :]

def hflip(img):
    """Horizontally flip the given image.
    Args:
        img (Numpy Image): Image to be flipped.
    Returns:
        Numpy Image:  Horizontall flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(img)))

    return img[..., ::-1, :]

def make_grid(tensor, grid_mode):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D Tensor of shape (64 x C x H x W)
            or a list of 64 images all of the same size.
        grid_mode: Type of grid. Should be: image or pixel. Default is image.
             - image: arrange the 64 images into a 8 x 8 grid of H x W pixels
             - pixel: arrange the 64 images into a H x W grid of 8 x 8 pixels
    Returns:
        Tensor: Grid Tensor image.
    """
    if not (torch.is_tensor(tensor) or
           (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    assert grid_mode in ['image', 'pixel'], \
        'Grid mode should be either image or pixel'

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        tensor = tensor.unsqueeze(0)

    if tensor.size(0) != 64:
        raise ValueError("Tensor should be of shape (64 x C x H x W) or a list of" +
            "64 images. Got {} images".format(tensor.size(0)))

    _, c, h, w = tensor.size()

    tensor = tensor.view(-1, 8, c, h, w).transpose_(0, 2).transpose_(1, 3)
    if grid_mode == 'image':
        tensor = tensor.transpose_(1, 2)
    elif grid_mode == 'pixel':
        tensor = tensor.transpose_(3, 4)
    tensor = tensor.reshape(c, 8 * h, 8 * w)

    return tensor.squeeze()

def concat(tensor):
    """Join a set of images along an existing axis.
    Args:
        tensor (Tensor or list): 4D Tensor of shape (B x C x H x W)
            or a list of B images all of the same size.
    Returns:
        Tensor: Grid Tensor image.
    """
    if not (torch.is_tensor(tensor) or
           (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        tensor = tensor.unsqueeze(0)

    _, _, h, w = tensor.size()

    return tensor.view(-1, h, w).squeeze()

