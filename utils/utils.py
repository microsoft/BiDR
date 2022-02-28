import torch.distributed as dist
import torch
import logging
import sys
import numpy as np
import random
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig
from lamb import Lamb

def is_main_process(local_rank):
    return local_rank in [-1, 0]


def setuplogging(level=logging.INFO):
    # silent transformers
    logging.getLogger("transformers").setLevel(logging.ERROR)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    if (root.hasHandlers()):
        root.handlers.clear()  # otherwise logging have multi output
    root.addHandler(handler)

def setup_worker(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def gather_vecs(tensor_list, broadcast_vecs, start_inx, end_inx):
    dist.all_gather(tensor_list, broadcast_vecs)
    all_vecs = torch.cat(tensor_list, dim=0)
    if start_inx is not None and end_inx is not None:
        all_vecs[start_inx:end_inx] = broadcast_vecs
    return all_vecs

def dist_gather_tensor(vecs, world_size, local_rank):
    all_tensors = [torch.empty_like(vecs) for _ in range(world_size)]
    dist.all_gather(all_tensors, vecs)
    all_tensors[local_rank] = vecs
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors

def create_optimizer_and_scheduler(args, model, num_training_steps: int):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.optimizer_str == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
    elif args.optimizer_str == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
    else:
        raise NotImplementedError("Optimizer must be adamw or lamb")

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    return optimizer, scheduler


def init_config(model_args, data_args, training_args):
    config = RobertaConfig.from_pretrained(
        'roberta-base',
        finetuning_task="msmarco",
        gradient_checkpointing=model_args.gradient_checkpointing)

    if model_args.num_hidden_layers != -1:
        config.num_hidden_layers = model_args.num_hidden_layers
    if model_args.output_embedding_size != -1:
        config.output_embedding_size = model_args.output_embedding_size
    else:
        config.output_embedding_size = config.hidden_size

    config.use_linear = training_args.use_linear
    config.fix_emb = training_args.fix_emb
    config.use_pq = training_args.use_pq
    config.partition = training_args.partition
    config.centroids = training_args.centroids
    config.init_index_path = training_args.init_index_path

    config.gradient_checkpointing = model_args.gradient_checkpointing

