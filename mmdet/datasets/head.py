from .custom import CustomDataset
from .registry import DATASETS
import pickle


@DATASETS.register_module
class HeadDataset(CustomDataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = ('head',)

    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as fp:
            img_infos = pickle.load(fp)

        return img_infos

    def get_ann_info(self, idx):
        ann_info = self.img_infos[idx]['ann']

        return ann_info
