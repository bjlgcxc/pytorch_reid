from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import math
import random

def func(op,img,magnitude):
    if op == "shearX":return shear(img, magnitude * 180, direction="x")
    elif op == "shearY":return shear(img, magnitude * 180, direction="y")
    elif op == "translateX":return img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude*img.size[0], 0, 1, 0), fillcolor=(128, 128, 128))
    elif op == "translateY":return img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude*img.size[1]), fillcolor=(128, 128, 128))
    elif op == "rotate":return img.rotate(magnitude)
    elif op=="color": return ImageEnhance.Color(img).enhance(magnitude)
    elif op =="posterize": return ImageOps.posterize(img, magnitude)
    elif op=="solarize": return ImageOps.solarize(img, magnitude)
    elif op =="contrast": return ImageEnhance.Contrast(img).enhance(magnitude)
    elif op=="sharpness": return ImageEnhance.Sharpness(img).enhance(magnitude)
    elif op=="brightness": return ImageEnhance.Brightness(img).enhance(magnitude)
    elif op=="autocontrast": return ImageOps.autocontrast(img)
    elif op=="equalize": return ImageOps.equalize(img)
    elif op=="invert": return ImageOps.invert(img)
    else: print('error ops')

class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self,keep_prob=0.5):
        self.keep_prob = keep_prob
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8),
        ]


    def __call__(self, img):
        if random.random() < self.keep_prob:
            return img
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2):
        ranges = {
            "shearX": np.linspace(-0.3, 0.3, 10),
            "shearY": np.linspace(-0.3, 0.3, 10),
            "translateX": np.linspace(-150 / 331, 150 / 331, 10),
            "translateY": np.linspace(-150 / 331, 150 / 331, 10),
            "rotate": np.linspace(-30, 30, 10),
            "color": np.linspace(0.1, 1.9, 10),
            "posterize": np.round(np.linspace(4, 8, 10), 0).astype(np.int),
            "solarize": np.linspace(0, 256, 10),
            "contrast": np.linspace(0.1, 1.9, 10),
            "sharpness": np.linspace(0.1, 1.9, 10),
            "brightness": np.linspace(0.1, 1.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }
        self.p1 = p1
        self.op1 = operation1
        self.op2 = operation2
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1: img = func(self.op1,img, self.magnitude1)
        if random.random() < self.p2: img = func(self.op2,img, self.magnitude2)
        return img



# from https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
def shear(img, angle_to_shear, direction="x"):
    width, height = img.size
    phi = math.tan(math.radians(angle_to_shear))

    if direction=="x":
        shift_in_pixels = phi * height

        if shift_in_pixels > 0:
            shift_in_pixels = math.ceil(shift_in_pixels)
        else:
            shift_in_pixels = math.floor(shift_in_pixels)

        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        # Note: PIL expects the inverse scale, so 1/scale_factor for example.
        transform_matrix = (1, phi, -matrix_offset, 0, 1, 0)

        img = img.transform((int(round(width + shift_in_pixels)), height),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC, fillcolor=(128, 128, 128))

    #     img = img.crop((abs(shift_in_pixels), 0, width, height))
        return img.resize((width, height), resample=Image.BICUBIC)

    elif direction == "y":
        shift_in_pixels = phi * width

        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        transform_matrix = (1, 0, 0, phi, 1, -matrix_offset)

        image = img.transform((width, int(round(height + shift_in_pixels))),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC, fillcolor=(128, 128, 128))

        # image = image.crop((0, abs(shift_in_pixels), width, height))

        return image.resize((width, height), resample=Image.BICUBIC)
