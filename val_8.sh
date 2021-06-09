python dense_retriever.py \
    model_file=/home/y247xie/01_exps/DPR/ckpts/nq_12348/dpr_biencoder.34.920 \
    qa_dataset=nq_test \
    ctx_datatsets=[dpr_wiki] \
    encoded_ctx_files=[\"/home/y247xie/01_exps/DPR/index/exp_test12348_test/index_*.pkl\"] \
    out_file=./predict_nq_12348_test

