import os
import json
import logging
import sys
import traceback
import random
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import IterableDataset
from collections import defaultdict
from dataset.dataset import SequenceDataset, load_rel

logger = logging.getLogger(__name__)


class DataloaderForSubGraphHard(IterableDataset):
    def __init__(self,
                 args,
                 rel_file,
                 rank_file,
                 queryids_cache, max_query_length,
                 docids_cache, max_doc_length,
                 local_rank,
                 world_size,
                 infer_path=None,
                 query_length=None,
                 doc_length=None,
                 enable_prefetch=True,
                 random_seed=42,
                 enable_gpu=True):
        '''
        Args:
        '''
        self.args = args
        self.query2pos = load_rel(rel_file)
        self.query_dataset = SequenceDataset(queryids_cache, max_query_length)
        self.doc_dataset = SequenceDataset(docids_cache, max_doc_length)
        self.docs_list = list(range(len(self.doc_dataset)))

        self.query2neg, self.neg2query = self.init_graph(rank_file, mink=args.mink, maxk=args.maxk,
                                                         n2_mink=args.n2_mink, n2_mak=args.n2_maxk,
                                                         query_set=set(list(self.query2pos.keys())))

        if len(self.query2pos) > len(self.query2neg):
            query2pos = {}
            for k,v in self.query2pos.items():
                if k in self.query2neg:
                    query2pos[k]=v
            self.query2pos = query2pos
            logging.info(f'*** *********please confirm the number of query in hardneg_path: {len(self.query2pos)}******************')

        assert args.per_query_hard_num > 0
        self.hard_num = args.per_query_hard_num
        self.query_length = query_length
        self.doc_length = doc_length

        self.local_rank = local_rank
        self.batch_size = world_size*args.per_device_train_batch_size
        self.enable_prefetch = enable_prefetch
        self.random_seed = random_seed
        self.enable_gpu = enable_gpu

        self.path_num = args.patch_num
        self.patch_batch_size = self.batch_size//self.path_num
        if args.fix_doc_emb:
            self.init_doc_embedding(infer_path, infer_path)

    def init_graph(self, rank_file, mink=0, maxk=200, n2_mink=0, n2_mak=200, query_set=None):
        rankdict = json.load(open(rank_file))
        print(f'loaded hardneg file:{rank_file}')
        query2neg = {}
        for k, v in rankdict.items():
            k = int(k)
            v = [int(_) for _ in v]
            v = v[mink:maxk]
            query2neg[k] = set(v)

        neg2query = defaultdict(set)
        for k, v in rankdict.items():
            k = int(k)
            v = [int(_) for _ in v]
            v = v[n2_mink:n2_mak]
            for neg in v:
                neg2query[neg].add(k)

        return query2neg, neg2query

    def __len__(self):
        return len(self.query2pos)

    def set_seed(self):
        random.seed(self.random_seed)
        self.random_seed += 1

    def init_query_set(self):
        self.query_set = list(self.query2pos.keys())
        self.set_seed()

        self.batch_num = len(self.query_set)//self.batch_size
        if len(self.query_set)%self.batch_size >= self.args.world_size:
            self.batch_num += 1
        self.query_set = set(self.query_set)

    def init_doc_embedding(self, doc_path, query_path):
        doc_embeddings = np.memmap(os.path.join(doc_path,"passages.memmap"),
                                   dtype=np.float32, mode="r")
        doc_ids = np.memmap(os.path.join(doc_path,"passages-id.memmap"),
                            dtype=np.int32, mode="r")
        self.doc_embeddings = doc_embeddings.reshape(-1, 768)
        self.doc_emb_index = {}
        for i, id in enumerate(doc_ids):
            self.doc_emb_index[id] = i

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        logging.info('__iter__')
        try:
            self.init_query_set()
            logging.info(f'{self.local_rank}, query num: {len(self.query_set)}')
            self.end = False
        except:
            error_type, error_value, error_trace = sys.exc_info()
            traceback.print_tb(error_trace)
            logging.info(error_value)
            self.pool.shutdown(wait=False)
            raise
        if self.enable_prefetch:
            self.start_async()
        else:
            self.outputs = self.generate_batch()
            self.outputs = self.outputs.__iter__()
        return self

    def start_async(self):
        logging.info(f'{self.local_rank}: start async...')
        self.aval_count = 0
        self.end = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def __next__(self):
        if self.enable_prefetch:
            if self.end and self.aval_count == 0:
                raise StopIteration
            next_batch = self.outputs.get()
            self.outputs.task_done()
            self.aval_count -= 1
            return next_batch
        else:
            next_data = self.outputs.__next__()
            if self.end:
                raise StopIteration
            return next_data

    def _produce(self):
        if self.enable_gpu:
            torch.cuda.set_device(self.local_rank)
        try:
            batch_gen = self.generate_batch()
            for batch in batch_gen:
                self.outputs.put(batch)
                self.aval_count += 1
        except:
            error_type, error_value, error_trace = sys.exc_info()
            traceback.print_tb(error_trace)
            logging.info(error_value)
            self.pool.shutdown(wait=False)
            raise

    def generate_batch(self):
        if self.args.generate_batch_method == 'random':
            return self.random_batch()
        elif self.args.generate_batch_method in ('dfs_subgraph', 'bfs_subgraph'):
            return self.subgraph_batch()
        else:
            raise NotImplementedError(f'{self.args.generate_batch_method}')

    def random_batch(self):
        for b in range(self.batch_num):
            assert len(self.query_set) != 0
            # self.set_seed()
            if len(self.query_set)>=self.batch_size:
                qids = random.sample(self.query_set, self.batch_size)
            else:
                qids = list(self.query_set)
            self.query_set = self.query_set - set(qids)
            yield self.get_data_by_qids(qids)
        self.end = True

    def get_data_by_qids(self, qids):
        queries_data = []
        docs_data = []
        hard_docs_data = []
        for qid in qids:
            pid = random.sample(self.query2pos[qid], 1)[0]
            q_data = self.query_dataset[qid]
            d_data = self.doc_dataset[pid]
            if self.query2neg is None:
                hardpids = random.sample(self.docs_list, self.hard_num)
            else:
                if self.hard_num > len(self.query2neg[qid]):
                    hardpids = random.sample(self.docs_list, self.hard_num-len(self.query2neg[qid]))
                    hardpids = hardpids + list(self.query2neg[qid])
                else:
                    hardpids = random.sample(self.query2neg[qid], self.hard_num)

            hard_d_data = [self.doc_dataset[hardpid] for hardpid in hardpids]
            queries_data.append(q_data)
            docs_data.append(d_data)
            hard_docs_data.extend(hard_d_data)
        return self.data_collate(queries_data, docs_data, hard_docs_data)

    def get_qid(self, search_method, cur_q_num):
        if cur_q_num%self.patch_batch_size==0:
            return random.sample(self.query_set, 1)[0]

        if len(self.node_queue) == 0:
            qid = random.sample(self.query_set, 1)[0]
            self.node_queue.append(qid)

        if search_method == 'dfs_subgraph':
            return self.node_queue.pop()
        elif search_method in ('bfs_subgraph'):
            return self.node_queue.pop(0)
        else:
            raise NotImplementedError

    def subgraph_batch(self):
        for b in range(self.batch_num):
            # self.set_seed()
            queries_data = []
            docs_data = []
            hard_docs_data = []
            qids_order = set()
            nids_order = set()
            self.node_queue = []

            while len(queries_data) < self.batch_size and len(self.query_set)>0:
                qid = self.get_qid(self.args.generate_batch_method, len(qids_order))
                while qid in qids_order:
                    qid = self.get_qid(self.args.generate_batch_method, len(qids_order))
                self.query_set.discard(qid)

                qids_order.add(qid)
                pid = random.sample(self.query2pos[qid], 1)[0]
                nids_order.add(pid)

                temp_neg_set = self.query2neg[qid] - nids_order
                if len(temp_neg_set) < self.hard_num:
                    rand_negs = random.sample(list(range(len(self.doc_dataset))), self.hard_num-len(temp_neg_set))
                    hardpids = list(temp_neg_set) + rand_negs
                else:
                    hardpids = random.sample(temp_neg_set, self.hard_num)

                temp_qs = []
                for hn in hardpids:
                    nids_order.add(hn)
                    for q in self.neg2query[hn]:
                        if q not in qids_order and q in self.query_set:
                            temp_qs.append(q)
                self.node_queue.extend(temp_qs)

                q_data = self.query_dataset[qid]
                d_data = self.doc_dataset[pid]
                hard_d_data = [self.doc_dataset[hardpid] for hardpid in hardpids]
                queries_data.append(q_data)
                docs_data.append(d_data)
                hard_docs_data.extend(hard_d_data)

            yield self.data_collate(queries_data, docs_data, hard_docs_data)
        self.end = True
        print(f'{self.local_rank}: end')


    @staticmethod
    def pack_tensor_2D(tokens_and_id, length=None):
        tokens_ids = [x['input_ids'] for x in tokens_and_id]
        mask = [x['attention_mask'] for x in tokens_and_id]
        item_ids = [x['id'] for x in tokens_and_id]

        length = length if length is not None else max(len(l) for l in tokens_ids)
        token_tensor = torch.zeros((len(tokens_ids), length), dtype=torch.int64)
        mask_tensor = torch.zeros((len(mask), length), dtype=torch.int64)
        for i, (t, m) in enumerate(zip(tokens_ids, mask)):
            token_tensor[i, :len(t)] = torch.tensor(t, dtype=torch.int64)
            mask_tensor[i, :len(t)] = torch.tensor(m, dtype=torch.int64)

        return token_tensor, mask_tensor, item_ids


    def data_collate(self, queries_data, docs_data, hard_docs_data):
        input_query_ids, query_attention_mask, qids = self.pack_tensor_2D(queries_data, length=self.query_length)
        input_doc_ids, doc_attention_mask, dids = self.pack_tensor_2D(docs_data, length=self.doc_length)
        neg_doc_ids, neg_doc_attention_mask, nids = self.pack_tensor_2D(hard_docs_data, length=self.doc_length)

        num_per_gpu = len(qids)//self.args.world_size
        all_num = num_per_gpu*self.args.world_size
        self.start_inx = num_per_gpu*self.local_rank
        self.end_inx = num_per_gpu*(1+self.local_rank)
        self.nstart_inx = num_per_gpu*self.local_rank*self.hard_num
        self.nend_inx = num_per_gpu*(self.local_rank+1)*self.hard_num

        qids = qids[:all_num]
        dids = dids[:all_num]
        nids = nids[:all_num*self.hard_num]

        rel_pair_mask = [[1 if docid not in self.query2pos[qid] else 0
                          for docid in dids]
                         for qid in qids]
        hard_pair_mask = [[1 if docid not in self.query2pos[qid] else 0
                           for docid in nids]
                          for qid in qids]
        rel_pair_mask = torch.FloatTensor(rel_pair_mask)
        hard_pair_mask = torch.FloatTensor(hard_pair_mask)

        q_emb, d_emb, n_emb = None, None, None
        if self.args.fix_doc_emb:
            if self.doc_emb_index is not None:
                doc_emb_inxs = [self.doc_emb_index[x] for x in dids]
                d_emb = self.doc_embeddings[doc_emb_inxs]
            else:
                d_emb = self.doc_embeddings[dids]
            d_emb = torch.FloatTensor(d_emb)

            if self.doc_emb_index is not None:
                doc_emb_inxs = [self.doc_emb_index[x] for x in nids]
                n_emb = self.doc_embeddings[doc_emb_inxs]
            else:
                n_emb = self.doc_embeddings[nids]
            n_emb = torch.FloatTensor(n_emb)

        batch_data = (
        input_query_ids[self.start_inx:self.end_inx], query_attention_mask[self.start_inx:self.end_inx],
        input_doc_ids[self.start_inx:self.end_inx], doc_attention_mask[self.start_inx:self.end_inx],
        neg_doc_ids[self.nstart_inx:self.nend_inx], neg_doc_attention_mask[self.nstart_inx:self.nend_inx],
        rel_pair_mask, hard_pair_mask, q_emb, d_emb, n_emb)
        if self.enable_gpu:
            batch_data = (x.cuda() if x is not None else None for x in batch_data)

        return batch_data


