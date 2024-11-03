import math
from torch.utils.data import Dataset as BaseDataset
from ..utilities import *


def load_data_from_path(path, idx, sec):
    print(sec)
    with open(path, "r") as f:
        lines = f.readlines()
    data = []
    data_index = []
    for i, line in enumerate(lines):
        f_path = line.rstrip()
        print(f_path)
        mix_path = '/l/users/fathinah.izzati/coco-mulla-repo/demo/input/tnj/'+f_path.split('/')[-1].replace('_emb','')
        data += [{"path": f_path,
                  "data": {
                      "rgb_emb":
                          np.load(f_path),
                        "mix": np.load(mix_path)
                  }}]
        data_index += [[i]]
    return data, data_index


class Dataset(BaseDataset):
    def __init__(self, path_lst, cfg, rid, sampling_prob=None, sampling_strategy=None, inference=False):
        super(Dataset, self).__init__()
        self.rid = rid
        self.rng = np.random.RandomState(42 + rid * 100)
        self.cfg = cfg
        self.data_index = []
        self.data  = []

        for i, path in enumerate(path_lst):
            data, data_index = load_data_from_path(path, i, cfg.sample_sec)
            print(data_index)
            self.data = data
            self.data_index += data_index

        self.f_len = len(self.data_index)

        self.epoch = 0
        self.f_offset = 0
        self.inference = inference

        self.descs = [
            "catchy song",
            "melodic music piece",
            "a song",
            "music tracks",
        ]
        self.sampling_strategy = sampling_strategy
        if sampling_prob is None:
            sampling_prob = [0., 0.8]
        self.sampling_prob = sampling_prob
        print("samling strategy", self.sampling_strategy, sampling_prob)

    def get_prompt(self):
        prompts = self.descs
        return prompts[self.rng.randint(len(prompts))]

    def load_data(self, set_id):
        data = self.data[set_id]
        return data["data"]


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        set_id = self.data_index[idx]
        data = self.load_data(set_id[0])
        mix = data["mix"]
        rgb_emb = data["rgb_emb"]

        cfg = self.cfg
        st = 0
        ed = 10
        desc = self.get_prompt()
        return {
             "mix": mix,
            "rgb_emb": rgb_emb,
            "cond_mask": np.ones(rgb_emb.shape),
            "desc": desc
        }

    def reset_random_seed(self, r, e):
        self.rng = np.random.RandomState(r + self.rid * 100)
        self.epoch = e
        self.rng.shuffle(self.data_index)


def collate_fn(batch):
    mix = torch.stack([torch.from_numpy(d["mix"]) for d in batch], 0)
    rgb_emb = torch.stack([torch.from_numpy(d["rgb_emb"]) for d in batch], 0)
    cond_mask = torch.stack([torch.from_numpy(d["cond_mask"]) for d in batch], 0)
    desc = [d["desc"] for d in batch]
    return {
        "mix": mix,
       "rgb_emb": rgb_emb,
        "cond_mask": cond_mask,
        "desc": desc,
    }
