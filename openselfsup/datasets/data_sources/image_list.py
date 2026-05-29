
import os
from PIL import Image

from ..registry import DATASOURCES
from .utils import McLoader


@DATASOURCES.register_module
class ImageList(object):

    def __init__(self, root, list_file, memcached=False, mclient_path=None, backend='mc', return_label=True):
        # print(root.split('/')[-1])
        if root.split('/')[-1] == 'images1w' or root.split('/')[-1] == 'miniImageNet':
            lines = os.listdir(root)
        else:
            with open(list_file, 'r') as f:
                lines = f.readlines()
        self.has_labels = len(lines[0].split()) == 2
        self.return_label = return_label
        if self.has_labels:
            self.fns, self.labels = zip(*[l.strip().split() for l in lines])
            self.labels = [int(l) for l in self.labels]
        else:
            # assert self.return_label is False
            self.fns = [l.strip() for l in lines]
        self.root = root
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False
        self.backend = backend

    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path, backend=self.backend)
            self.initialized = True

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        name = self.fns[idx].split('/')[-1]
        if self.memcached:
            self._init_memcached()
            img = self.mc_loader.get_item(self.fns[idx])
        else:
            img = Image.open(self.fns[idx])

        img = img.convert('RGB')
        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return img, target
        else:
            return img, name
