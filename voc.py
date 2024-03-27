import os
from PIL import Image
from torch.utils import data

num_classes = 21
ignore_label = 255
root = './data'

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''


#Feel free to convert this palette to a map
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]  #3 values- R,G,B for every class. First 3 values for class 0, next 3 for
#class 1 and so on......
palette_map = {
    0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128], 5: [128, 0, 128], 6: [0, 128, 128],
    7: [128, 128, 128], 8: [64, 0, 0], 9: [192, 0, 0], 10: [64, 128, 0], 11: [192, 128, 0], 12: [64, 0, 128],
    13: [192, 0, 128], 14: [64, 128, 128], 15: [192, 128, 128], 16: [0, 64, 0], 17: [128, 64, 0], 18: [0, 192, 0],
    19: [128, 192, 0], 20: [0, 64, 128]
}


def make_dataset(mode):
    """
    Creates a list of tuples (image_path, mask_path) for a given dataset mode.

    Asserts that the mode is one of 'train', 'val', or 'test'.
    Based on the mode, it reads corresponding image and mask paths from the VOC dataset.
    For each image, it pairs its path with the corresponding mask path.

    TODO: Make similar for val and test set.

    Args:
        mode (str): The mode of the dataset, either 'train', 'val', or 'test'.

    Returns:
        list of tuples: Each tuple contains paths (image_path, mask_path).
    """
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
        
    return items


class VOC(data.Dataset):
    """
    A custom dataset class for VOC dataset.
    Maintain the structure of this so that it is easily compatible with Pytorch's dataloader.

    - Resizes images and masks to a specified width and height.
    - Implements methods to get dataset items and dataset length.

    - TIP: You may add an additional argument for common transformation for both the image and mask
           to help with data augmentation in part 4.c

    Args:
        mode (str): Mode of the dataset ('train', 'val', etc.).
        transform (callable, optional): Transform to be applied to the images.
        target_transform (callable, optional): Transform to be applied to the masks.
    """

    def __init__(self, mode, transform=None, target_transform=None, common_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.common_transform = common_transform
        self.width = 224
        self.height = 224

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB').resize((self.width, self.height))
        mask = Image.open(mask_path).resize((self.width, self.height))

        if self.common_transform is not None:
            img, mask = self.common_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        mask[mask==ignore_label]=0

        return img, mask

    def __len__(self):
        return len(self.imgs)