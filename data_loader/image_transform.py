"""Image preprocessing shared by train.py, test.py and the caching tools (EXP-009)."""
import torchvision.transforms as transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PREPROCESS_CHOICES = ['raw', 'clip']


def image_transform(preprocess='raw'):
    """raw: Resize 224 + ToTensor, as in the original code (no normalisation).
    clip: bicubic Resize 224 + ToTensor + Normalize with CLIP's mean/std. This is CLIP's own
    preprocess for square images and also LAVIS's BLIP-2 eval processor (same mean/std, image size 224)."""
    if preprocess == 'raw':
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    if preprocess == 'clip':
        return transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                   transforms.ToTensor(),
                                   transforms.Normalize(CLIP_MEAN, CLIP_STD)])
    raise ValueError(preprocess)
