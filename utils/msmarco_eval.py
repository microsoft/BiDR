"""
This is official eval script opensourced on MSMarco site (not written or owned by us)

This module computes evaluation metrics for MSMARCO dataset on the ranking task.
Command line:
python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>

Creation Date : 06/12/2018
Last Modified : 1/21/2019
Authors : Daniel Campos <dacamp@microsoft.com>, Rutger van Haasteren <ruvanh@microsoft.com>
"""
"""
I (Jingtao Zhan) modified this script for evaluating MSMARCO Doc dataset. --- 4/19/2021
"""
import sys
import statistics

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




def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, path_to_candidate=None):
    """Compute MRR metric
    Args:    
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR = 0
    MRR_100 = 0
    NDCG, Recall = 0, 0
    Recall1, Recall5, Recall20, Recall50, Recall1000, Recall5000,  Recall10000, Recall20000, Recall3000, Recall50000, Recall30 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    Recall10 = 0
    Recall200, Recall300, Recall500 = 0.0, 0.0, 0.0
    qids_with_relevant_passages = 0
    ranking = []

    fout = None
    if path_to_candidate is not None:
        fout = open(path_to_candidate+'_details_scores', 'w')

    for qid in qids_to_ranked_candidate_passages:
        temp_score = {}
        if qid in qids_to_relevant_passageids:

            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]

            # assert len(target_pid) == 1
            actual_y = [0] * MaxMRRRank
            # m = 0
            temp_score['MRR10'] = 0.0
            temp_score['MRR100'] = 0.0
            for i in range(0, 10):
                if candidate_pid[i] in target_pid:
                    actual_y[i] = 1
                    MRR += 1/(i + 1)
                    temp_score['MRR10'] = 1/(i+1)
                    ranking.pop()
                    ranking.append(i+1)
                    break

            for i in range(0,100):
                if candidate_pid[i] in target_pid:
                    MRR_100 += 1/(i + 1)
                    temp_score['MRR100'] = 1/(i+1)
                    # ranking.pop()
                    # ranking.append(i+1)
                    break

            temp_score['R10'] = (len(set.intersection(set(target_pid), set(candidate_pid[:10])))/len(set(target_pid)))
            temp_score['R50'] = (len(set.intersection(set(target_pid), set(candidate_pid[:50])))/len(set(target_pid)))
            temp_score['R100'] = (len(set.intersection(set(target_pid), set(candidate_pid[:100])))/len(set(target_pid)))
            temp_score['R1000'] = (len(set.intersection(set(target_pid), set(candidate_pid[:1000])))/len(set(target_pid)))
            if fout:
                fout.write(json.dumps(temp_score)+'\n')

            # print(len(set.intersection(set(target_pid), set(candidate_pid[:MaxMRRRank]))))
            Recall1 += (len(set.intersection(set(target_pid), set(candidate_pid[:1])))/len(set(target_pid)))
            Recall5 += (len(set.intersection(set(target_pid), set(candidate_pid[:5])))/len(set(target_pid)))
            Recall20 += (len(set.intersection(set(target_pid), set(candidate_pid[:20])))/len(set(target_pid)))
            Recall30 += (len(set.intersection(set(target_pid), set(candidate_pid[:30])))/len(set(target_pid)))
            Recall50 += (len(set.intersection(set(target_pid), set(candidate_pid[:50])))/len(set(target_pid)))
            Recall10 += (len(set.intersection(set(target_pid), set(candidate_pid[:10])))/len(set(target_pid)))
            Recall200 += (len(set.intersection(set(target_pid), set(candidate_pid[:200])))/len(set(target_pid)))
            Recall300 += (len(set.intersection(set(target_pid), set(candidate_pid[:300])))/len(set(target_pid)))
            Recall500 += (len(set.intersection(set(target_pid), set(candidate_pid[:500])))/len(set(target_pid)))
            Recall1000 += (len(set.intersection(set(target_pid), set(candidate_pid[:1000]))) / len(set(target_pid)))

            Recall += (len(set.intersection(set(target_pid), set(candidate_pid[:100])))/len(set(target_pid)))
            if len(candidate_pid)>=10000:
                Recall5000 += (len(set.intersection(set(target_pid), set(candidate_pid[:5000])))/len(set(target_pid)))
                Recall10000 += (len(set.intersection(set(target_pid), set(candidate_pid[:10000])))/len(set(target_pid)))
                # Recall20000 += (len(set.intersection(set(target_pid), set(candidate_pid[:20000])))/len(set(target_pid)))
                Recall3000 += (len(set.intersection(set(target_pid), set(candidate_pid[:3000])))/len(set(target_pid)))
                # Recall50000 += (len(set.intersection(set(target_pid), set(candidate_pid[:50000])))/len(set(target_pid)))

            best_y = [0]*(MaxMRRRank)
            best_y[0] = 1
            # for i in range(len(set.intersection(set(target_pid), set(candidate_pid)))):
            #     if i<MaxMRRRank:
            #         best_y[i] = 1
            best = dcg_score(np.array(best_y))
            actual = dcg_score(np.array(actual_y))
            NDCG += (actual/best)



    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
    
    MRR = MRR/len(qids_to_relevant_passageids)
    MRR_100 = MRR_100/len(qids_to_relevant_passageids)
    Recall = Recall/len(qids_to_relevant_passageids)
    NDCG = NDCG/len(qids_to_relevant_passageids)
    all_scores[f'MRR @{MaxMRRRank}'] = MRR
    all_scores[f'MRR @100'] = MRR_100
    # all_scores[f'NDCG @{MaxMRRRank}'] = NDCG
    all_scores[f'Recall @10'] = Recall10/len(qids_to_relevant_passageids)
    all_scores[f'Recall @50'] = Recall50/len(qids_to_relevant_passageids)
    all_scores[f'Recall @100'] = Recall
    all_scores[f'Recall @200'] = Recall200/len(qids_to_relevant_passageids)
    all_scores[f'Recall @300'] = Recall300/len(qids_to_relevant_passageids)
    all_scores[f'Recall @500'] = Recall500/len(qids_to_relevant_passageids)
    all_scores[f'Recall @1'] = Recall1/len(qids_to_relevant_passageids)
    all_scores[f'Recall @5'] = Recall5/len(qids_to_relevant_passageids)
    all_scores[f'Recall @20'] = Recall20/len(qids_to_relevant_passageids)
    all_scores[f'Recall @30'] = Recall30/len(qids_to_relevant_passageids)
    all_scores[f'Recall @1000'] = Recall1000/len(qids_to_relevant_passageids)
    all_scores[f'Recall @10000'] = Recall10000/len(qids_to_relevant_passageids)
    all_scores[f'Recall @20000'] = Recall20000/len(qids_to_relevant_passageids)
    all_scores[f'Recall @3000'] = Recall3000/len(qids_to_relevant_passageids)
    # all_scores[f'Recall @50000'] = Recall50000/len(qids_to_relevant_passageids)
    all_scores[f'Recall @5000'] = Recall5000/len(qids_to_relevant_passageids)

    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores
                
def compute_metrics_from_files(path_to_reference, path_to_candidate, perform_checks=True, output_details=False):
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

    if output_details:
        return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, path_to_candidate)

    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, None)

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