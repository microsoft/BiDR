# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
sys.path.append('./')
import argparse
import logging
import os
import numpy as np
from utils.msmarco_eval import compute_metrics_from_files

logger = logging.Logger(__name__)

def load_q_k_score(file, topk=10000):
    q_k_score = {}
    for line in open(file, 'r', encoding='utf-8'):
        try:
            q, k, rank, score = line.strip('\n').split('\t')
            q, k, rank, score = int(q), int(k), int(rank), float(score)
        except:
            print(line)
            # exit(0)
        if q not in q_k_score:
            q_k_score[q] = ([], [])
        if len(q_k_score[q][0]) < topk:
            q_k_score[q][0].append(k)
            q_k_score[q][1].append(score)
    return q_k_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", choices=["passage", 'doc', 'trec_doc', 'quora'], type=str, required=True)
    parser.add_argument("--mode", type=str,
                        choices=["train", "dev", "test", "lead", "test2019", "test2020"], default='dev')
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--candidate_from_ann", type=str, required=True)
    parser.add_argument("--sparse_weight", type=float, required=True, default=0.3)
    parser.add_argument("--output_embedding_size", type=int, required=True, default=768)
    parser.add_argument("--root_output_dir", type=str, required=False, default='./data')

    parser.add_argument("--doc_file", type=str, required=False, default=None)
    parser.add_argument("--query_file", type=str, required=False, default=None)

    parser.add_argument("--MRR_cutoff", type=int, default=10)
    parser.add_argument("--Recall_cutoff", type=int, nargs='+', default=[5, 10, 30, 50, 100])

    args = parser.parse_args()
    args.output_dir = os.path.join(f"{args.root_output_dir}/{args.data_type}/", args.output_dir)
    args.output_score_file = os.path.join(args.output_dir, f"{args.mode}.dense_score.tsv")
    args.output_rank_file = os.path.join(args.output_dir, f"{args.mode}.post_verifiaction.tsv")

    if args.query_file is None:
        args.query_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query.memmap")
        args.queryids_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query-id.memmap")
    else:
        args.query_memmap_path = os.path.join(args.query_file, f"{args.mode}-query.memmap")
        args.queryids_memmap_path = os.path.join(args.query_file, f"{args.mode}-query-id.memmap")

    if args.doc_file is None:
        args.doc_memmap_path = os.path.join(args.output_dir, "passages.memmap")
        args.docid_memmap_path = os.path.join(args.output_dir, "passages-id.memmap")
    else:
        args.doc_memmap_path = os.path.join(args.doc_file, "passages.memmap")
        args.docid_memmap_path = os.path.join(args.doc_file, "passages-id.memmap")

    doc_embeddings = np.memmap(args.doc_memmap_path,
                               dtype=np.float32, mode="r")
    doc_ids = np.memmap(args.docid_memmap_path,
                        dtype=np.int32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, args.output_embedding_size)

    query_embeddings = np.memmap(args.query_memmap_path,
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, args.output_embedding_size)

    query_ids = np.memmap(args.queryids_memmap_path,
                          dtype=np.int32, mode="r")

    q_k_score = load_q_k_score(args.candidate_from_ann, args.topk)

    docid2inx = {}
    for i, id in enumerate(doc_ids):
        docid2inx[id] = i
    qid2inx = {}
    for i, id in enumerate(query_ids):
        qid2inx[id] = i

    print('predicing----------------------')
    with open(args.output_score_file, 'w') as f:
        count = 0
        for q in q_k_score.keys():
            ks, scores = q_k_score[q]

            qinx = [qid2inx[q]]
            ksinx = [docid2inx[x] for x in ks]
            q_emb = query_embeddings[qinx]
            ks_emb = doc_embeddings[ksinx]

            rank_score = np.matmul(q_emb, ks_emb.T)[0]
            for k, s2 in zip(ks, rank_score):
                f.write(f"{q}\t{k}\t-1\t{s2}\n")
        count += 1
        if count % 500 == 0:
            print(f'---{count}---')

    rerank_q_k_score = load_q_k_score(args.output_score_file)
    final_q_k_score = dict()
    for q in q_k_score.keys():
        ks1, scores1 = q_k_score[q]
        ks2, scores2 = rerank_q_k_score[q]
        temp_s = []
        for i, (s1, s2) in enumerate(zip(scores1,scores2)):
            assert ks1[i] == ks2[i]
            s = s2 + args.sparse_weight * s1
            temp_s.append(s)
        final_q_k_score[q] = (ks1, temp_s)


    outputfile = open(args.output_rank_file, 'w', encoding='utf-8')
    for q in final_q_k_score.keys():
        ks, final_score = final_q_k_score[q]
        sorted_inx = np.argsort(final_score)[::-1]
        sorted_ks = np.array(ks)[sorted_inx]

        for idx, k in enumerate(sorted_ks):
            outputfile.write(f"{q}\t{k}\t{idx + 1}\t{final_score[idx]}\n")

    if args.mode != 'train':
        path_to_reference = f'./data/{args.data_type}/preprocess/{args.mode}-qrel.tsv'
        metrics = compute_metrics_from_files(path_to_reference, args.output_rank_file, args.MRR_cutoff, args.Recall_cutoff)
        print(f'#####################{args.output_rank_file}: ')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))

if __name__ == "__main__":
    main()
