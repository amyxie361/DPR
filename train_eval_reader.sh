python -m torch.distributed.launch --nproc_per_node=2 \
    train_extractive_reader.py \
    encoder.sequence_length=350 \
    train_files=/data/y247xie/01_exps/DPR/predict_nq_train_12346 \
    dev_files=/data/y247xie/01_exps/DPR/predict_nq_dev_12346 \
    gold_passages_src=/data/y247xie/01_exps/DPR/nq_train.json \
    gold_passages_src_dev=/data/y247xie/01_exps/DPR/nq_dev.json \
    output_dir=/data/y247xie/01_exps/DPR/outputs/12346_nq_reader \
    train.batch_size=1 \
    train.dev_batch_size=1 \
    seed=42 \
    train.learning_rate=1e-5 \
    train.eval_step=30000 \
    do_lower_case=True \
    eval_top_docs=[20] \
    train.warmup_steps=0 \
    train.num_train_epochs=20




