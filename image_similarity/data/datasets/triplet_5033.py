import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.error import URLError
from urllib.request import urlretrieve

from alive_progress import alive_bar
from torch.utils.data import Dataset

from image_similarity.util.dir import get_project_root


@dataclass
class Triplet:
    search_term: str
    images: List[str]


class TripletsDataset(Dataset):
    """
    Class to load a triplets dataset from a triplets URLs source file.
    The expected format is as follows:
    ```triplets.txt
    <search_term_1>
    <search_term_1:positive_example_image_url>
    <search_term_1:query_image_url>
    <search_term_1:negative_example_image_url>
    <search_term_2>
    <search_term_2:positive_example_image_url>
    <search_term_2:query_image_url>
    <search_term_2:negative_example_image_url>
    ...
    ```
    """
    DEFAULT_SOURCEFILE_PATH = "resources/triplet_5033/triplet_5033.txt"
    DEFAULT_DATA_PATH = "resources/triplet_5033/data"

    def __init__(
        self, sourcefile_path: Optional[str] = None, data_path: Optional[str] = None
    ):
        self.sourcefile_path = sourcefile_path or get_project_root().joinpath(
            self.DEFAULT_SOURCEFILE_PATH
        )
        self.data_path = Path(
            data_path or get_project_root().joinpath(self.DEFAULT_DATA_PATH)
        )
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.samples = self._get_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def _get_samples(self) -> List:
        unloaded_triplets = self._read_triplets()
        return self._load_triplets(unloaded_triplets)

    def _read_triplets(self) -> List[Triplet]:
        logging.info("Reading triplets file...")
        triplets = []
        with open(self.sourcefile_path) as sourcefile:
            while True:
                search_term = sourcefile.readline()
                if not search_term:
                    break
                triplet_images = [
                    file.strip() for file in
                    (sourcefile.readline() for _ in range(3)) if file
                ]
                if len(triplet_images) != 3:
                    raise RuntimeError(
                        f"Unable to read incomplete triplet:"
                        f" query='{search_term}' triplet={triplet_images}"
                    )
                triplets.append(Triplet(search_term, triplet_images))
        logging.info(f"{len(triplets)} triplets read")
        return triplets

    def _load_triplets(self, unloaded_triplets: List[Triplet]) -> List:
        logging.info("Loading triplets...")
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
        logging.info(f"{len(triplets)} of {len(unloaded_triplets)} triplets sucessfully loaded")
        logging.info(f"Dataset totals {len(triplets) * 3} images")
        return triplets
