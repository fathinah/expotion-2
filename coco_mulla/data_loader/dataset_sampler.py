import math
from torch.utils.data import Dataset as BaseDataset
from ..utilities import *
import numpy as np
from scipy.interpolate import CubicSpline



# Original time points (video sequence length)
t_original = np.linspace(0, 1, 50)
# New time points (target sequence length)
t_new = np.linspace(0, 1, 501)

def interpolate(data):
    data_new = np.zeros((501, 512))
    for i in range(512):  # Iterate over features
        cs = CubicSpline(t_original, data[:, i])
        data_new[:, i] = cs(t_new)
    return data_new

def load_data_from_path(path, idx, sec):
    with open(path, "r") as f:
        lines = f.readlines()
    data = []
    data_index = []
    for i, line in enumerate(lines):
        line = line.rstrip()
        f_path = line.split(" ")[0]
        onset = float(line.split(" ")[1])
        offset = float(line.split(" ")[2])
        video = np.load(os.path.join(f_path, "video_emb.npy"))
        data += [{"path": f_path,
                  "data": {
                      "video":interpolate(video)
                  }}]
        onset = math.ceil(onset)
        offset = int(offset)
        data_index += [[idx, i, j] for j in range(onset, offset, 10)]
    return data, data_index


class Dataset(BaseDataset):
    def __init__(self, path_lst, cfg, rid, sampling_prob=None, sampling_strategy=None, inference=False):
        super(Dataset, self).__init__()
        self.rid = rid
        self.rng = np.random.RandomState(42 + rid * 100)
        self.cfg = cfg
        self.data = []
        self.data_index = []

        for i, path in enumerate(path_lst):
            print(i,path)
            data, data_index = load_data_from_path(path, i, cfg.sample_sec)
            self.data.append(data)
            self.data_index += data_index

        self.f_len = len(self.data_index)
        print("num of files", self.f_len)

        self.epoch = 0
        self.f_offset = 0
        self.inference = inference

        self.descs = [
            "A realistic and high quality soundtrack and sound effect for the video",
            "High quality sountrack for a cartoon movie",
            "A realistic and high quality soundtrack and sound effect for the video",
            "High quality sountrack for a cartoon movie"
            
        ]

    def get_prompt(self):

        prompts = self.descs
        return prompts[self.rng.randint(len(prompts))]

    def load_data(self, set_id, song_id):
        data = self.data[set_id]
        if "music" not in data[song_id]["data"]:
            video = data[song_id]["data"]["video"]
            music_path = os.path.join(data[song_id]["path"], "music_emb.npy")

            music = np.load(music_path)
            result = {
                "music": music,
                "video":video,
            }
            data[song_id]["data"] = result
        return data[song_id]["data"]



    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        set_id, sid, sec_id = self.data_index[idx]
        data = self.load_data(set_id, sid)
        music = data["music"]
        video = data["video"]
        pad_frame = np.zeros((1, video.shape[1]), dtype=video.dtype)
        video = np.concatenate([video, pad_frame], axis=0)
        cfg = self.cfg
        st = sec_id
        ed = st + cfg.sample_sec
        frame_st = int(st * cfg.frame_res)
        frame_ed = int(ed * cfg.frame_res)

        music = music[:, frame_st: frame_ed]
        video = video[frame_st: frame_ed+1, : ]
        print('music', music.shape)
        print('video', video.shape)
        desc = self.get_prompt()
        return {
            "music":music,
            "video": video,
            # "videmb":videmb,
            "desc": desc
        }
    def reset_random_seed(self, r, e):
        self.rng = np.random.RandomState(r + self.rid * 100)
        self.epoch = e
        self.rng.shuffle(self.data_index)

def collate_fn(batch):
    music = torch.stack([torch.from_numpy(d["music"]) for d in batch], 0)
    video = torch.stack([torch.from_numpy(d["video"]) for d in batch], 0)
    desc = [d["desc"] for d in batch]
    return {
        "music": music,
        "video": video,
        "desc": desc,
    }