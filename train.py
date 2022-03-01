# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import logging
import os
import time
from pathlib import Path
import traceback

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
from transformers import HfArgumentParser

from arguments import DataTrainingArguments, ModelArguments, MyTrainingArguments
from utils.utils import setup_worker, setuplogging, create_optimizer_and_scheduler, init_config, dist_gather_tensor
from dataset.dataloader import DataloaderForSubGraphHard
from dataset.dataset import TextTokenIdsCache
from model.model import RobertaDotTrainModel

logger = logging.Logger(__name__)


def train(local_rank, model_args, data_args, training_args):
    try:
        setuplogging()
        logger.info(f"Training/evaluation parameters {training_args}")

        os.environ["RANK"] = str(local_rank)
        setup_worker(local_rank, training_args.world_size)
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu',
                              local_rank)

        config = init_config(model_args, data_args, training_args)
        model = RobertaDotTrainModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config
        )
        if model_args.pq_path is not None:
            pq_ckpt = torch.load(os.path.join(model_args.pq_path, 'pytorch_model.bin'),map_location='cpu')
            pq_ckpt = {k: v for k, v in pq_ckpt.items() if k in ('rotate', 'codebook')}
            print('loading...', pq_ckpt.keys())
            model.load_state_dict(pq_ckpt, strict=False)
        model.to(device)
        ddp_model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True)

        dataloader = DataloaderForSubGraphHard(
            args=training_args,
            rel_file=os.path.join(data_args.data_dir, "train-qrel.tsv"),
            rank_file=data_args.hardneg_json,
            queryids_cache=TextTokenIdsCache(data_dir=data_args.data_dir, prefix="train-query"),
            max_query_length=data_args.max_query_length,
            docids_cache=TextTokenIdsCache(data_dir=data_args.data_dir, prefix="passages"),
            max_doc_length=data_args.max_doc_length,
            local_rank=local_rank,
            world_size=training_args.world_size,
            enable_prefetch=True,
            random_seed=training_args.seed,
            enable_gpu=True,
            infer_path=training_args.infer_path
        )

        all_step_num = len(dataloader)*training_args.num_train_epochs//(training_args.per_device_train_batch_size*training_args.world_size)
        optimizer, scheduler = create_optimizer_and_scheduler(training_args, ddp_model, all_step_num)

        global_step = 0
        start_time = time.time()

        loss, match_loss, match_quant_loss, l2_loss = 0.0, 0.0, 0.0, 0.0
        for ep in range(int(training_args.num_train_epochs)):
            ddp_model.train()
            for step, sample in enumerate(dataloader):
                input_query_ids, query_attention_mask, \
                input_doc_ids, doc_attention_mask, \
                neg_doc_ids, neg_doc_attention_mask, \
                rel_pair_mask, hard_pair_mask, \
                q_emb, k_emb, n_emb = (x.cuda(device=device, non_blocking=True) if x is not None else None for x in sample)

                q_vecs = ddp_model.module.query_emb(input_query_ids, query_attention_mask)
                if not training_args.fix_doc_emb:
                    k_vecs = ddp_model.module.body_emb(input_doc_ids, doc_attention_mask)
                    quant_k = ddp_model.module.quant(k_vecs) if training_args.use_pq else None
                    n_vecs = ddp_model.module.body_emb(neg_doc_ids, neg_doc_attention_mask) if neg_doc_ids is not None else None
                    quant_n = ddp_model.module.quant(n_vecs) if training_args.use_pq else None
                else:
                    k_vecs = k_emb.detach()
                    n_vecs = n_emb.detach()
                    quant_k = ddp_model.module.quant(k_vecs) if training_args.use_pq else None
                    quant_n = ddp_model.module.quant(n_vecs) if training_args.use_pq else None

                if training_args.world_size > 1:
                    q_vecs = dist_gather_tensor(q_vecs, training_args.world_size, local_rank)
                    k_vecs = dist_gather_tensor(k_vecs, training_args.world_size, local_rank)
                    n_vecs = dist_gather_tensor(n_vecs, training_args.world_size, local_rank)
                    if training_args.use_pq:
                        quant_k = dist_gather_tensor(quant_k, training_args.world_size, local_rank)
                        quant_n = dist_gather_tensor(quant_n, training_args.world_size, local_rank)

                batch_loss, batch_match_loss, batch_match_quant_loss, batch_l2loss = ddp_model(q_vecs, k_vecs, n_vecs,
                                       rel_pair_mask=rel_pair_mask, hard_pair_mask=hard_pair_mask,
                                       loss_method=training_args.loss_method,
                                       temperature=training_args.temperature,
                                       hard_k=quant_k, hard_n=quant_n, quant_weight=training_args.quantloss_weight)

                loss += batch_loss.item()
                match_loss += batch_match_loss.item()
                if not isinstance(batch_match_quant_loss, float):
                    match_quant_loss += batch_match_quant_loss.item()
                    l2_loss += batch_l2loss.item()

                batch_loss.backward()
                if training_args.max_grad_norm != -1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

                if (step + 1) % training_args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    optimizer.zero_grad()

                    if global_step % training_args.logging_steps == 0:
                        logging.info('[{}] step:{}, lr:{}, train_loss: {:.5f} = {:.5f} + {:.5f} + {:.5f}'.
                                     format(local_rank,
                                            global_step,
                                            optimizer.param_groups[0]['lr'],
                                            loss / training_args.logging_steps,
                                            match_loss/training_args.logging_steps,
                                            match_quant_loss/training_args.logging_steps,
                                            l2_loss/training_args.logging_steps))
                        loss, match_loss, match_quant_loss, l2_loss = 0.0, 0.0, 0.0, 0.0

                    if global_step % training_args.save_steps == 0 and local_rank == 0:
                        ckpt_path = os.path.join(data_args.save_model_path,
                                                 f'{data_args.savename}/{global_step}/')
                        Path(ckpt_path).mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(ckpt_path, 'pytorch_model.bin'))
                        config.to_json_file( os.path.join(ckpt_path, 'config.json'))
                        logging.info(f"Model saved to {ckpt_path}")
                dist.barrier()

            logging.info("train time:{}".format(time.time() - start_time))
            if local_rank == 0 and ep%1 == 0:
                ckpt_path = os.path.join(data_args.save_model_path,
                                         f'{data_args.savename}/{ep}/')
                Path(ckpt_path).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(ckpt_path,'pytorch_model.bin'))
                config.to_json_file(os.path.join(ckpt_path, 'config.json'))
                logging.info(f"Model saved to {ckpt_path}")
            dist.barrier()
    except:
        error_type, error_value, error_trace = sys.exc_info()
        traceback.print_tb(error_trace)
        logging.info(error_value)


def main():
    setuplogging()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    gpus = ','.join(training_args.gpu_rank.split('_'))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = training_args.master_port

    logging.info(training_args)
    logging.info(model_args)
    logging.info(data_args)

    if training_args.world_size == 1:
        train(0, model_args, data_args, training_args)
    else:
        mp.spawn(train,
                 args=(model_args, data_args, training_args),
                 nprocs=training_args.world_size,
                 join=True)

if __name__ == "__main__":
    main()