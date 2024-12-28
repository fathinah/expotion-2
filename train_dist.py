import os
import torch
from coco_mulla.models import CoCoMulla
from coco_mulla.utilities import *
from torch.nn.parallel import DistributedDataParallel as DDP

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

    # train(replica_id, model, dataset, dataloader, device, model_dir,
    #       args.learning_rate)
