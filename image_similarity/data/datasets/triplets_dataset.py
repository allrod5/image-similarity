import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable, Union
from urllib.error import URLError
from urllib.request import urlretrieve
from PIL import Image

import numpy as np
import torch
from alive_progress import alive_bar
from skimage import color, img_as_ubyte, transform
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms

from image_similarity.util.dir import get_project_root

count = 0

@dataclass
class Triplet:
    search_term: str
    images: List[str]

    @property
    def positive_image(self):
        return self.images[0]

    @property
    def query_image(self):
        return self.images[1]

    @property
    def negative_image(self):
        return self.images[3]

    def read_images(self) -> List[np.ndarray]:
        return [imread(image) for image in self.images]


class TripletsDataset(Dataset):
    """
    Class to load a triplets dataset from a triplets URLs source file.
    The following is the expected format:
    ```triplets.txt
    <search_term_1:search_string>
    <search_term_1:positive_example_image_url>
    <search_term_1:query_image_url>
    <search_term_1:negative_example_image_url>
    <search_term_2:search_string>
    <search_term_2:positive_example_image_url>
    <search_term_2:query_image_url>
    <search_term_2:negative_example_image_url>
    ...
    ```
    """

    DEFAULT_SOURCEFILE_PATH = "resources/triplet_5033/triplet_5033.txt"
    DEFAULT_DATA_PATH = "resources/triplet_5033/data"

    def __init__(
        self,
        sourcefile_path: Optional[str] = None,
        data_path: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.sourcefile_path = sourcefile_path or get_project_root().joinpath(
            self.DEFAULT_SOURCEFILE_PATH
        )
        self.data_path = Path(
            data_path or get_project_root().joinpath(self.DEFAULT_DATA_PATH)
        )
        self.transform = transform or transforms.Compose(
            [
                self.NormalizeToRGBA(),
                self.Rescale(256),
                self.RandomCrop(224),
                self.ToTensor(),
            ]
        )
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.samples = self._get_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: Union[int, torch.Tensor]):
        if torch.is_tensor(index):
            index = index.tolist()

        triplet = self.samples[index]
        sample = triplet.read_images()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_samples(self) -> List[Triplet]:
        triplets = self._read_triplets()
        return self._download_triplets(triplets)

    def _read_triplets(self) -> List[Triplet]:
        logging.info("Reading triplets file...")
        triplets = []
        with open(self.sourcefile_path) as sourcefile:
            while True:
                search_term = sourcefile.readline()
                if not search_term:
                    break
                triplet_images = [
                    file.strip()
                    for file in (sourcefile.readline() for _ in range(3))
                    if file
                ]
                if len(triplet_images) != 3:
                    raise RuntimeError(
                        f"Unable to read incomplete triplet:"
                        f" query='{search_term}' triplet={triplet_images}"
                    )
                triplets.append(Triplet(search_term, triplet_images))
        logging.info(f"{len(triplets)} triplets read")
        return triplets

    def _download_triplets(self, unloaded_triplets: List[Triplet]) -> List:
        logging.info("Downloading triplets...")
        triplets = []
        with alive_bar(len(unloaded_triplets) * 3, force_tty=True) as progress_bar:
            for triplet in unloaded_triplets:
                try:
                    images = []
                    for url in triplet.images:
                        file_path = Path(self.data_path.joinpath(url.replace("/", " ")))
                        if not file_path.exists():
                            urlretrieve(url, file_path)
                        images.append(file_path)
                        progress_bar()
                    triplets.append(Triplet(triplet.search_term, images))
                except URLError as e:
                    logging.error(e)
                    for _ in range(3):
                        progress_bar()
        logging.info(
            f"{len(triplets)} of {len(unloaded_triplets)} triplets sucessfully downloaded"
        )
        logging.info(f"Dataset totals {len(triplets) * 3} images")
        return triplets

    class NormalizeToRGBA:
        def __call__(self, sample):
            normalized_sample = []
            for image in sample:
                normalized = image
                if len(image.shape) == 2:
                    normalized = color.gray2rgb(image)
                elif image.shape[2] == 4:
                    normalized = color.rgba2rgb(image)
                # if len(image.shape) == 2:
                #     normalized = color.gray2rgb(image, alpha=True)
                # elif image.shape[2] == 3:
                #     normalized = np.dstack((image, np.zeros([image.shape[0], image.shape[1]])))
                normalized_sample.append(normalized)
            return normalized_sample

    class Rescale(object):
        """Rescale the image in a sample to a given size.

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
        """

        def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            self.output_size = output_size

        def __call__(self, sample):
            return [self._rescale_image(image) for image in sample]

        def _rescale_image(self, image):
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)
            resized_image = img_as_ubyte(transform.resize(image, (new_h, new_w)))
            return resized_image

    class RandomCrop(object):
        """Crop randomly the image in a sample.

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """

        def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size

        def __call__(self, sample):
            return [self._crop_image(image) for image in sample]

        def _crop_image(self, image):
            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            cropped_image = image[top: top + new_h, left: left + new_w]
            return cropped_image

    class ToTensor(object):
        """Convert ndarrays in sample to Tensors."""

        def __call__(self, sample: List[np.ndarray]):
            # swap color axis using transpose because
            # numpy image: H x W x C
            # torch image: C X H X W
            bla = [
                torch.from_numpy(image.transpose((2, 0, 1))).view(1, 3, 224, 224).float()
                for image in sample
            ]
            return bla
