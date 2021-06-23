import csv
import json
import sys
import os
from utils_qa import normalize_answer, f1_score, exact_match_score, metric_max_over_ground_truths
from collections import Counter

def nf_f1score(old_prediction, new_prediction, ground_truth): # todo
    old_prediction_tokens = normalize_answer(old_prediction).split()
    new_prediction_tokens = normalize_answer(new_prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    new_pred = Counter(new_prediction_tokens)
    old_pred = Counter(old_prediction_tokens)
    gt_pred = Counter(ground_truth_tokens)

    old_common = old_pred & gt_pred
    new_common = new_pred & gt_pred
    old_new_common = old_pred & new_pred

    three_common =  new_pred & old_pred & gt_pred
    union = new_pred | old_pred | gt_pred

    old_miss = new_common - three_common
    new_miss = old_common - three_common
    both_fp = old_new_common - three_common

    new_fp = new_pred - new_common - both_fp
    old_fp = old_pred - old_common - both_fp
    both_miss = gt_pred - old_common - old_miss

    returns = {
        "ttt": 1.0 * sum(three_common.values()) / sum(union.values()),
        "tft": 1.0 * sum(old_miss.values()) / sum(union.values()),
        "ttf": 1.0 * sum(new_miss.values()) / sum(union.values()),
        "ftt": 1.0 * sum(both_fp.values()) / sum(union.values()),
        "fft": 1.0 * sum(new_fp.values()) / sum(union.values()),
        "ftf": 1.0 * sum(old_fp.values()) / sum(union.values()),
        "tff": 1.0 * sum(both_miss.values()) / sum(union.values()),
    }

    # num_same = sum(common.values())
    # if num_same == 0:
    #     return 0
    # precision = 1.0 * num_same / len(prediction_tokens)
    # recall = 1.0 * num_same / len(ground_truth_tokens)
    # f1 = (2 * precision * recall) / (precision + recall)
    return returns["ttf"] + returns["fft"]


def nf_metric_min_over_ground_truths(old_prediction, new_prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = nf_f1score(old_prediction, new_prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return min(scores_for_ground_truths)

def evalutate_model_pair(dataset_file, old_model_predict_file, new_model_predict_file, task="SQUAD1"):
    with open(old_model_predict_file) as prediction_file:
        old_predictions = json.load(prediction_file)
    with open(new_model_predict_file) as prediction_file:
        new_predictions = json.load(prediction_file)
    if task == "SQUAD1":
        with open(dataset_file) as dataset_file:
            dataset_json = json.load(dataset_file)
        # if dataset_json["version"] != expected_version:
        #     print(
        #         "Evaluation expects v-" + expected_version + ", but got dataset with v-" + dataset_json["version"],
        #         file=sys.stderr,
        #     )
            dataset = dataset_json["data"]
    elif task == "NQ":
        dataset = old_predictions

    old_f1 = old_exact_match = new_f1 = new_exact_match = total = 0
    nf_f1 = nf_exact_match = pf_f1 = pf_exact_match = 0

    if task == "SQUAD1":

        for article in dataset:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    total += 1

                    ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                # qa["id"]
                    if (qa["id"] not in old_predictions) and (qa["id"] not in new_predictions):
                        message = "Unanswered question " + qa["id"] + " will receive score 0."
                        print(message, file=sys.stderr)
                        continue
                    elif qa["id"] not in old_predictions:
                        new_prediction = new_predictions[qa["id"]]
                        old_prediction = ""
                    elif qa["id"] not in new_predictions:
                        old_prediction = old_predictions[qa["id"]]
                        new_prediction = ""
                    else:
                        new_prediction = new_predictions[qa["id"]]
                        old_prediction = old_predictions[qa["id"]]


                    old_exact_match_ = metric_max_over_ground_truths(exact_match_score, old_prediction, ground_truths)
                    old_f1_ = metric_max_over_ground_truths(f1_score, old_prediction, ground_truths)
                    new_exact_match_ = metric_max_over_ground_truths(exact_match_score, new_prediction, ground_truths)
                    new_f1_ = metric_max_over_ground_truths(f1_score, new_prediction, ground_truths)

                    old_exact_match += old_exact_match_
                    old_f1 += old_f1_
                    new_exact_match += new_exact_match_
                    new_f1 += new_f1_

                    if old_exact_match_ > new_exact_match_:
                        nf_exact_match += (old_exact_match_ - new_exact_match_)
                    elif new_exact_match_ > old_exact_match_:
                        pf_exact_match += (new_exact_match_ - old_exact_match_)

                    nf_f1 += nf_metric_min_over_ground_truths(old_prediction, new_prediction, ground_truths)
                    pf_f1 += nf_metric_min_over_ground_truths(new_prediction, old_prediction, ground_truths)

    elif task == "NQ":
        total = len(old_predictions)
        gold_dict = {}
        old_dict = {}
        new_dict = {}
        for q in old_predictions:
            question = q["question"]
            ground_truths = q["gold_answers"]
            if question not in gold_dict:
                gold_dict[question] = ground_truths
                old_dict[question] = q
            else:
                print("duplic question", question, ground_truths)
        for q in new_predictions:
            question = q["question"]
            ground_truths = q["gold_answers"]
            if question in gold_dict:
                if gold_dict[question] != ground_truths:
                    print("conflict golden answer, ", question, ground_truths)
                else:
                    new_dict[question] = q
            else:
                gold_dict[question] = ground_truths
                new_dict[question] = q
            

        for q in gold_dict:
            ground_truths = gold_dict[q]
            if q in old_dict:
                old_prediction = old_dict[q]["predictions"][0]["prediction"]["text"]
            else:
                old_prediction = ""
            if q in new_dict:
                new_prediction = new_dict[q]["predictions"][0]["prediction"]["text"]
            else:
                new_prediction = ""
            old_exact_match_ = metric_max_over_ground_truths(exact_match_score, old_prediction, ground_truths)
            old_f1_ = metric_max_over_ground_truths(f1_score, old_prediction, ground_truths)
            new_exact_match_ = metric_max_over_ground_truths(exact_match_score, new_prediction, ground_truths)
            new_f1_ = metric_max_over_ground_truths(f1_score, new_prediction, ground_truths)
            old_exact_match += old_exact_match_
            old_f1 += old_f1_
            new_exact_match += new_exact_match_
            new_f1 += new_f1_
            if old_exact_match_ > new_exact_match_:
                nf_exact_match += (old_exact_match_ - new_exact_match_)
            elif new_exact_match_ > old_exact_match_:
                pf_exact_match += (new_exact_match_ - old_exact_match_)
            nf_f1 += nf_metric_min_over_ground_truths(old_prediction, new_prediction, ground_truths)
            pf_f1 += nf_metric_min_over_ground_truths(new_prediction, old_prediction, ground_truths)





    old_exact_match = 100.0 * old_exact_match / total
    new_exact_match = 100.0 * new_exact_match / total
    old_f1 = 100.0 * old_f1 / total
    new_f1 = 100.0 * new_f1 / total
    nf_exact_match = 100.0 * nf_exact_match / total
    pf_exact_match = 100.0 * pf_exact_match / total
    nf_f1 = 100.0 * nf_f1 / total
    pf_f1 = 100.0 * pf_f1 / total

    return {
        "old_exact_match": old_exact_match,
        "new_exact_match": new_exact_match,
        "old_f1": old_f1,
        "new_f1": new_f1 ,
        "nf_exact_match": nf_exact_match,
        "pf_exact_match": pf_exact_match,
        "nf_f1": nf_f1,
        "pf_f1": pf_f1}

if __name__ == "__main__":

# model1_dir = "./baseline_all"
# model1 = [f.path for f in os.scandir(model1_dir) if f.is_dir() ]

# model2_dir = "train_track"
# model2 = [f.path for f in os.scandir(model2_dir) if f.is_dir()]

# task = "SQUAD"
# for seed in [0, 1, 2, 3, 4]:
#     model1 = "/home/ubuntu/00_exps/pytorch-elmo-classification/exps_new/epoch5_lr3e-4_batch32-drop0.2-elmo-dub_initseed{}_shufseed{}".format(seed, seed)
#     for model_new in model2:
#         if "SEED" + str(seed) in model_new :
#             r = evaluate_model_pair(model1, model_new, tasks)[task]
#             print("\t".join(model_new.split("_")[-6:]) + "\t{:.4f}\t{:.4f}".format(r[0], r[1]))
#             print(model_new, r)
    task = "NQ"

    #model1 = "outputs/12345_reader_inf_2.36946"
    #model2 = "outputs/12347_reader_inf_2.50856"
    model1 = "outputs/12347_reader_inf_2.50856"
    model2 = "outputs/12345_reader_inf_2.36946"
    gt = "dev-v1.1.json"

    print(evalutate_model_pair(gt, model1, model2, task=task))
