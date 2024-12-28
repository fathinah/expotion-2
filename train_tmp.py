import argparse
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from coco_mulla.utilities.trainer_utils import Trainer

import torch
import torch.nn as nn
import os
from config import TrainCfg
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm

from coco_mulla.data_loader.dataset_sampler import Dataset, collate_fn
from coco_mulla.models import CoCoMulla

device = "cuda"
N_GPUS = 4


def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]


def get_dataset(rid, dataset_split, sampling_strategy, sampling_prob):

    file_lst = ["train.lst",
                "train.lst"]
    splits = [
        [1],
        [0],
        [0, 1],
    ]
    dataset = Dataset(
        rid=rid,
        path_lst=[file_lst[i] for i in splits[dataset_split]],
        sampling_prob=sampling_prob,
        sampling_strategy=sampling_strategy,
        cfg=TrainCfg)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=TrainCfg.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
        sampler=DistributedSampler(dataset),
        pin_memory=True,
        drop_last=True)

    return dataset, dataloader


def train_dist(replica_id, replica_count, port, model_dir, args):
    print('masuk')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = CoCoMulla(TrainCfg.sample_sec, num_layers=args.num_layers, latent_dim=args.latent_dim).to(device)
    model.set_training()
    model = DDP(model, [replica_id])
    dataset, dataloader = get_dataset(rid=replica_id, dataset_split=args.dataset,
                                      sampling_strategy=args.sampling_strategy,
                                      sampling_prob=[args.sampling_prob_a, args.sampling_prob_b])

    train(replica_id, model, dataset, dataloader, device, model_dir,
          args.learning_rate)


def loss_fn(outputs, y):
    prob = outputs.logits
    mask = outputs.mask
    prob = prob[mask]
    y = y[mask]
    prob = prob.view(-1, 2048)
    return nn.CrossEntropyLoss()(prob, y)


def train(rank, model, dataset, dataloader, device, model_dir, learning_rate):
    # optimizer and lr scheduler
    num_steps = len(dataloader)
    epochs = TrainCfg.epoch
    rng = np.random.RandomState(569 + rank * 100)
    if rank == 0:
        writer = SummaryWriter(model_dir, flush_secs=20)

    trainer = Trainer(params=model.parameters(), lr=learning_rate, num_epochs=epochs, num_steps=num_steps)

    model = model.to(device)
    step = 0
    for e in range(0, epochs):
        mean_loss = 0
        n_element = 0
        model.train()

        dl = tqdm(dataloader, desc=f"Epoch {e}") if rank == 0 else dataloader
        r = rng.randint(0, 233333)
        dataset.reset_random_seed(r, e)
        for i, batch in enumerate(dl):
            desc = batch["desc"]
            mix = batch["mix"].to(device).long()
            drums = batch["drums"].to(device).long()
            chords = batch["chords"].to(device).float()
            piano_roll = batch["piano_roll"].to(device).float()
            cond_mask = batch["cond_mask"].to(device).long()

            batch_1 = {
                "seq": mix,
                "drums": drums,
                "chords": chords,
                "piano_roll": piano_roll,
                "cond_mask": cond_mask,
                "desc": desc,

            }
            # with autocast:
            outputs = model(**batch_1)
            r_loss = loss_fn(outputs, mix.long())

            grad_1, lr_1 = trainer.step(r_loss, model.parameters())

            step += 1
            n_element += 1
            if rank == 0:
                writer.add_scalar("r_loss", r_loss.item(), step)
                writer.add_scalar("grad_1", grad_1, step)
                writer.add_scalar("lr_1", lr_1, step)

            mean_loss += r_loss.item()

        mean_loss = mean_loss / n_element
        if rank == 0:
            with torch.no_grad():
                writer.add_scalar('train/mean_loss', mean_loss, step)
                model.module.save_weights(os.path.join(model_dir, f"diff_{e}_end.pth"))


def main(args):
    experiment_folder = args.experiment_folder
    experiment_name = args.experiment_name

    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    model_dir = os.path.join(experiment_folder, experiment_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    world_size = N_GPUS
    port = _get_free_port()
    spawn(train_dist, args=(world_size, port, model_dir, args), nprocs=world_size, join=True)


# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('-d', '--experiment_folder', type=str)
#     # parser.add_argument('-n', '--experiment_name', type=str)
#     # parser.add_argument('-l', '--num_layers', type=int)
#     # parser.add_argument('-t', '--text_path', type=str, default=None)
#     # parser.add_argument('-r', '--latent_dim', type=int)
#     # parser.add_argument('-lr', '--learning_rate', type=float)
#     # parser.add_argument('-s', '--sampling_strategy', type=str)
#     # parser.add_argument('-a', '--sampling_prob_a', type=float, default=0.)
#     # parser.add_argument('-b', '--sampling_prob_b', type=float, default=0.)
#     # parser.add_argument('-ds', '--dataset', type=int, default=0)

#     # args = parser.parse_args()
#     # main(args)
#     args = {
#         "num_layers": 48,
#         "latent_dim": 12,
#         "experiment_folder": "/l/users/fathinah.izzati/coco-mulla-repo/expe",
#         "experiment_name": "experiment_1",
#         "prompt_path": "/l/users/fathinah.izzati/coco-mulla-repo/demo/input/let_it_be.prompt.txt",
#         'sampling_strategy':'prob-based',
#         "dataset": '/l/users/fathinah.izzati/coco-mulla-repo/train.lst',
#         'learning_rate':0.1

#     }
#     args = SimpleNamespace(**args)
#     main(args)
