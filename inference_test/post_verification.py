# coding=utf-8
import argparse
import sys
import logging
import os
import numpy as np
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
                        choices=["train", "dev", "test", "lead", "dev1", "dev2", "test2019", "test2020"], required=True)
    parser.add_argument("--topk", type=int, default=10000)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, default='evaluate/star')
    parser.add_argument("--q_k_score_file", type=str, required=True, default='evaluate/star')
    parser.add_argument("--ensemble_weigth", type=float, required=True, default=1.0)
    parser.add_argument("--ori_weight", type=float, required=True, default=1.0)
    parser.add_argument("--rerank_num", type=int, required=True, default=200)
    parser.add_argument("--output_embedding_size", type=int, required=True, default=768)

    parser.add_argument("--root_output_dir", type=str, required=False, default='./data')
    parser.add_argument("--gpu_rank", type=str, required=False, default='0_1_2_3_4_5_6_7')
    parser.add_argument("--output_score", type=int, required=True, default=0)

    parser.add_argument("--doc_file", type=str, required=False, default=None)
    parser.add_argument("--query_file", type=str, required=False, default=None)


    args = parser.parse_args()

    gpus = ','.join(args.gpu_rank.split('_'))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    args.output_dir = os.path.join(f"{args.root_output_dir}/{args.data_type}/", args.output_dir)
    if args.query_file is None:
        args.query_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query.memmap")
        args.queryids_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query-id.memmap")
    else:
        args.query_memmap_path = os.path.join(args.query_file, f"{args.mode}-query.memmap")
        args.queryids_memmap_path = os.path.join(args.query_file, f"{args.mode}-query-id.memmap")



    args.output_rank_file = os.path.join(args.output_dir, f"{args.mode}.rank_{args.topk}_rerank_score.tsv")

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

    print('topk', args.topk)
    if args.q_k_quant_score_file is not None:
        q_k_score = load_q_k_score(args.q_k_quant_score_file, args.topk)
    else:
        q_k_score = load_q_k_score(args.q_k_score_file, args.topk)

    docid2inx = {}
    for i, id in enumerate(doc_ids):
        docid2inx[id] = i
    qid2inx = {}
    for i, id in enumerate(query_ids):
        qid2inx[id] = i


    print(args.output_rank_file)
    # if not os.path.exists(args.output_rank_file):
    print('predicing----------------------')
    with open(args.output_rank_file, 'w') as f:
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

    rerank_q_k_score = load_q_k_score(args.output_rank_file)
    final_q_k_score = dict()
    for q in q_k_score.keys():
        ks1, scores1 = q_k_score[q]
        ks2, scores2 = rerank_q_k_score[q]
        temp_s = []
        # print(ks1[:10], ks2[:10])
        for i, (s1, s2) in enumerate(zip(scores1,scores2)):
            assert ks1[i] == ks2[i]
            if 'hnsw' in args.q_k_quant_score_file:
                # print(s1, 1-s1)
                s =  args.ensemble_weigth * s2 + args.ori_weight * (1-s1)
            else:
                s =  args.ensemble_weigth * s2 + args.ori_weight * (s1)
            temp_s.append(s)
        final_q_k_score[q] = (ks1, temp_s)


    rerank_nums = [100, 200, 300, 500, 1000, 3000, 5000, 8000, 10000]
    if '50000' in args.q_k_score_file:
        rerank_nums = [100, 1000, 5000, 10000, 20000, 30000, 50000]
    if args.topk==1000:
        rerank_nums = [100, 300, 500, 1000]


    outputfile = []
    for file_idx, rerank_num in enumerate(rerank_nums):
        outputfile.append(open(args.output_rank_file + f'_{rerank_num}', 'w', encoding='utf-8'))
        print(args.output_rank_file + f'_{rerank_num}')


    for q in final_q_k_score.keys():
        ks, final_score = final_q_k_score[q]
        for file_idx, rerank_num in enumerate(rerank_nums):
            temp_final_score = final_score[:rerank_num]
            temp_ks = ks[:rerank_num]

            sorted_inx = np.argsort(temp_final_score)[::-1]
            sorted_ks = np.array(temp_ks)[sorted_inx]

            for idx, k in enumerate(sorted_ks):
                if args.output_score == 1:
                    outputfile[file_idx].write(f"{q}\t{k}\t{idx + 1}\t{final_score[idx]}\n")
                else:
                    outputfile[file_idx].write(f"{q}\t{k}\t{idx + 1}\n")




if __name__ == "__main__":
    main()
