# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from collections import Counter
import numpy as np
import json

MaxMRRRank = 10
EVAL_DOC = False

def load_reference_from_stream(f):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    qids_to_relevant_passageids = {}
    for l in f:
        try:
            if EVAL_DOC:
                l = l.strip().split(' ')
            else:
                l = l.strip().split('\t')
            qid = int(l[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            if EVAL_DOC:
                assert l[2][0] == "D"
                if int(l[3])>=1:
                    qids_to_relevant_passageids[qid].append(int(l[2][1:]))
            else:
                if int(l[3])>=1:
                    qids_to_relevant_passageids[qid].append(int(l[2]))
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids

def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    with open(path_to_reference,'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids

def load_candidate_from_stream(f):
    """Load candidate data from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    qid_to_ranked_candidate_passages = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            qid = int(l[0])
            if EVAL_DOC:
                assert l[1][0] == "D"
                pid = int(l[1][1:])
            else:
                pid = int(l[1])
            rank = int(l[2])
            if qid in qid_to_ranked_candidate_passages:
                pass    
            else:
                # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                tmp = [0] * 20000
                qid_to_ranked_candidate_passages[qid] = tmp
            qid_to_ranked_candidate_passages[qid][rank-1]=pid
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qid_to_ranked_candidate_passages
                
def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    
    with open(path_to_candidate,'r') as f:
        qid_to_ranked_candidate_passages = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_passages

def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Perform quality checks on the dictionaries

    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set([item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1])

        if len(duplicate_pids-set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                    qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message

def dcg_score(y_true):
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)




def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, MRR_cutoff, Recall_cutoff):
    """Compute MRR metric
    Args:    
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    MRR = 0.0
    Recall = [0.0]*len(Recall_cutoff)
    ranking = []
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:

            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]

            for i in range(0, MRR_cutoff):
                if candidate_pid[i] in target_pid:
                    MRR += 1/(i + 1)
                    ranking.pop()
                    ranking.append(i+1)
                    break

            for i, k in enumerate(Recall_cutoff):
                Recall[i] += (len(set.intersection(set(target_pid), set(candidate_pid[:k])))/len(set(target_pid)))

    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
    
    MRR = MRR/len(qids_to_relevant_passageids)
    Recall = [x/len(qids_to_relevant_passageids) for x in Recall]

    print(f'MRR@{MRR_cutoff}:{MRR}')
    for i, k in enumerate(Recall_cutoff):
        print(f'Recall@{k}:{Recall[i]}')

    return MRR, Recall
                
def compute_metrics_from_files(path_to_reference, path_to_candidate, MRR_cutoff, Recall_cutoff, perform_checks=True):
    """Compute MRR metric
    Args:    
    p_path_to_reference_file (str): path to reference file.
        Reference file should contain lines in the following format:
            QUERYID\tPASSAGEID
            Where PASSAGEID is a relevant passage for a query. Note QUERYID can repeat on different lines with different PASSAGEIDs
    p_path_to_candidate_file (str): path to candidate file.
        Candidate file sould contain lines in the following format:
            QUERYID\tPASSAGEID1\tRank
            If a user wishes to use the TREC format please run the script with a -t flag at the end. If this flag is used the expected format is 
            QUERYID\tITER\tDOCNO\tRANK\tSIM\tRUNID 
            Where the values are separated by tabs and ranked in order of relevance 
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    
    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '': print(message)

    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, MRR_cutoff, Recall_cutoff)

def main():
    """Command line:
    python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>
    """
    print("Eval Started")
    if len(sys.argv) in [3,4] :
        path_to_reference = sys.argv[1]
        path_to_candidate = sys.argv[2]
        if len(sys.argv) == 4:
            global MaxMRRRank
            if sys.argv[3] == "doc":
                global EVAL_DOC
                MaxMRRRank, EVAL_DOC = 100, True
            else:
                MaxMRRRank = int(sys.argv[3])
        metrics = compute_metrics_from_files(path_to_reference, path_to_candidate)
        print('#####################')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
        print('#####################')

    else:
        print('Usage: msmarco_eval_ranking.py <reference ranking> <candidate ranking> [MaxMRRRank or DocEval]')
        exit()
    
if __name__ == '__main__':
    main()