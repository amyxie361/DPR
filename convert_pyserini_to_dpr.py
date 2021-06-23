import json
import pickle
import tqdm
data = json.load(open("outputs/run.dpr.nq-train.bm25.k10.9-b0.4.json",'r'))

res = []

def have_answer(text, answers):
    for ans in answers:
        if ans in text:
            return True
    return False

for key in tqdm.tqdm(data):
    res.append({
        "question": data[key]["question"],
        "answers": data[key]["answers"],
        "ctxs": [{
            "id":x["docid"], 
            "title":x["text"].split("\n")[0], 
            "text":x["text"].split("\n")[1], 
            "score":x["score"], 
            "has_answer": have_answer(x["text"].split("\n")[1], data[key]["answers"])
            } for x in data[key]["contexts"]],
        })

json.dump(res, open("/data/y247xie/01_exps/DPR/outputs/run.dpr.nq-train.bm25.k10.9-b0.4",'w'))


