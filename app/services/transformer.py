import numpy as np
import torch
from fvcore.transforms.transform import Transform
from torch import Tensor
from typing import Dict, Optional, Union, Tuple
from abc import ABCMeta, abstractmethod
import inspect
import pprint

# Config 파일에서 정의된 값들을 참조
CONFIG = {
    "RESIZE_IMG": (256, 256),
    "TEMPORAL_SCALING": 1.0,
    "CROP_SIZE": (224, 224),
    "TEMPORAL_CROP_RATIO": 0.0,
    "MEAN": (0.0637, 0.0988, 0.2312),
    "STD": (0.0643, 0.0556, 0.1150)
}

def normalize(tensor, mean, std):
    if not torch.is_tensor(tensor):
        raise TypeError('tensor should be a torch tensor. Got {}.'.format(type(tensor)))
    assert tensor.ndim in (3, 4), (
        "Expected tensor to be a tensor image of size (C, H, W) or (N, C, H, W)."
        "Got tensor.size() ={}.".format(tensor.size())
    )
    dtype = tensor.dtype
    device = tensor.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'.
            format(dtype)
        )
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    if tensor.ndim == 4:
        mean = mean[None, :, ...]
        std = std[None, :, ...]
    tensor.sub_(mean).div_(std)
    return tensor

def extract_img_size(img):
    if len(img.shape) == 4:
        t, h, w = img.shape[:3]
    elif len(img.shape) in (2, 3):
        h, w = img.shape[:2]
        t = None
    else:
        raise ("Unsupported input with shape of {}".format(img.shape))
    return t, h, w

def to_tensor(
    numpy_array: np.ndarray,
    *,
    normalizer: Optional[Dict] = None,
    divider: Union[int, float] = 255.0,
    is_5d_tensor: bool = False
) -> Tensor:
    assert isinstance(numpy_array, np.ndarray)
    ndim = len(numpy_array.shape)
    assert ndim in (3, 4)
    numpy_array = numpy_array / (divider if numpy_array.dtype == np.uint8 else 1)
    if is_5d_tensor:
        assert ndim == 4
        shift_factor = 4
    else:
        shift_factor = 3
    numpy_array = np.moveaxis(numpy_array, -1, ndim - shift_factor).astype(np.float32)
    if is_5d_tensor:
        numpy_array = numpy_array[None, ...]
    float_tensor = torch.from_numpy(np.ascontiguousarray(numpy_array))
    if normalizer is not None:
        return normalize(float_tensor, **normalizer)
    return float_tensor

def to_numpy(float_tensor: torch.Tensor, target_dtype: np.dtype) -> np.ndarray:
    assert float_tensor.ndim == 5
    assert float_tensor.size(0) == 1
    float_tensor = float_tensor.squeeze(0).permute(1, 2, 3, 0)
    if target_dtype == np.uint8:
        float_tensor = float_tensor.round().byte()
    return float_tensor.numpy()


class ToTensor(Transform):
    def __init__(self, normalizer=None, target_dtype=torch.float32):
        super().__init__()
        self.normalizer = normalizer
        self.target_dtype = target_dtype

    def apply_image(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(img)} instead.")

        if img.dtype != np.uint8:
            print(f"Warning: unexpected image dtype {img.dtype}. Expected uint8.")

        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)

        if self.normalizer:
            tensor = normalize(tensor, self.normalizer['mean'], self.normalizer['std'])

        return tensor.to(self.target_dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def inverse(self) -> Transform:
        pass


class ScaleTransform(Transform):
    def __init__(
        self,
        h: int,
        w: int,
        new_h: int,
        new_w: int,
        *,
        t: Optional[int] = None,
        new_t: Optional[int] = None,
        interp: str = None
    ):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: Optional[str] = None) -> np.ndarray:
        if len(img.shape) == 4:
            t, h, w = img.shape[:3]
        elif len(img.shape) in (2, 3):
            h, w = img.shape[:2]
        else:
            raise ("Unsupported input with shape of {}".format(img.shape))
        assert (self.h == h and self.w == w), "Input size mismatch h w {}:{} -> {}:{}".format(self.h, self.w, h, w)
        interp_method = interp if interp is not None else self.interp
        if interp_method in ["linear", "bilinear", "trilinear", "bicubic"]:
            align_corners = False
        else:
            align_corners = None
        if self.t is not None and self.new_t is not None:
            new_size = (self.new_t, self.new_h, self.new_w)
        else:
            new_size = (self.new_h, self.new_w)
        float_tensor = torch.nn.functional.interpolate(
            to_tensor(img, divider=1, is_5d_tensor=True),
            size=new_size,
            mode=interp_method,
            align_corners=align_corners,
        )
        return to_numpy(float_tensor, img.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        pass

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        pass

    def inverse(self) -> Transform:
        pass


class CropTransform(Transform):
    def __init__(
        self,
        x0: int,
        y0: int,
        w: int,
        h: int,
        *,
        t0: Optional[int] = None,
        z: Optional[int] = None
    ):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) <= 3:
            return img[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        else:
            if self.t0 is not None and self.z is not None:
                return img[self.t0:self.t0 + self.z, self.y0:self.y0 + self.h, self.x0:self.x0 + self.w, :]
            return img[..., self.y0:self.y0 + self.h, self.x0:self.x0 + self.w, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        pass

    def apply_polygons(self, polygons: list) -> list:
        pass


class TransformGen(metaclass=ABCMeta):
    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def get_transform(self, img):
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__

class Resize(TransformGen):
    def __init__(self, shape=CONFIG['RESIZE_IMG'], *, temporal_scaling: Optional[float] = CONFIG['TEMPORAL_SCALING'], interp="bicubic"):
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, img):
        t, h, w = extract_img_size(img)
        new_t = None
        if self.temporal_scaling is not None:
            assert t is not None
            new_t = int(t * self.temporal_scaling)
        return ScaleTransform(
            h, w, self.shape[0], self.shape[1], t=t, new_t=new_t, interp=self.interp
        )


class RandomCrop(TransformGen):
    def __init__(self, crop_type: str = "absolute", crop_size=CONFIG['CROP_SIZE'], *, temporal_crop_ratio: Optional[float] = CONFIG['TEMPORAL_CROP_RATIO']):
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute"]
        self._init(locals())

    def get_transform(self, img):
        t, h, w = extract_img_size(img)
        croph, cropw = self.get_crop_size((h, w))
        if self.temporal_crop_ratio is not None:
            assert t is not None
            cropt = int(t * (1. - self.temporal_crop_ratio) + 0.5)
            assert t >= cropt
        else:
            cropt = None
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        t0 = np.random.randint(t - cropt + 1) if cropt is not None else None
        return CropTransform(w0, h0, cropw, croph, t0=t0, z=cropt)

    def get_crop_size(self, image_size):
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return self.crop_size
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


class CenterCrop(TransformGen):
    def __init__(self, crop_size: Tuple[int, int] = CONFIG['CROP_SIZE'], *, temporal_crop_ratio: Optional[float] = CONFIG['TEMPORAL_CROP_RATIO']):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        t, h, w = extract_img_size(img)
        croph, cropw = self.crop_size
        if self.temporal_crop_ratio is not None:
            assert t is not None
            cropt = int(t * (1. - self.temporal_crop_ratio) + 0.5)
            assert t >= cropt
        else:
            cropt = None
        h0 = int(0.5 * (h - croph))
        w0 = int(0.5 * (w - cropw))
        t0 = int(0.5 * (t - cropt)) if cropt is not None else None
        return CropTransform(w0, h0, cropw, croph, t0=t0, z=cropt)


class ToTensorGen(TransformGen):
    def __init__(self, normalizer=None):
        super().__init__()
        self.normalizer = normalizer

        # `normalizer` 값이 존재할 경우, 실수형으로 변환
        if self.normalizer:
            self.normalizer = {
                "mean": [float(m) for m in self.normalizer["mean"]],
                "std": [float(s) for s in self.normalizer["std"]]
            }

    def get_transform(self, img):
        target_dtype = torch.float32  # 모든 텐서를 float32로 변환
        return ToTensor(normalizer=self.normalizer, target_dtype=target_dtype)