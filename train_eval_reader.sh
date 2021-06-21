python train_extractive_reader.py \
    encoder.sequence_length=350 \
    train_files=/home/y247xie/01_exps/DPR/outputs/predict_nq_train_12347 \
    dev_files=/home/y247xie/01_exps/DPR/outputs/predict_nq_dev_12347 \
    gold_passages_src=/home/y247xie/01_exps/DPR/resource/downloads/data/gold_passages_info/nq_train.json \
    gold_passages_src_dev=/home/y247xie/01_exps/DPR/resource/downloads/data/gold_passages_info/nq_dev.json \
    output_dir=/home/y247xie/01_exps/DPR/outputs/12347_nq_reader_new \
    train.batch_size=1 \
    train.dev_batch_size=1\
    +train.save_step=2000 \
    seed=42 \
    train.learning_rate=1e-5 \
    train.eval_step=10000 \
    do_lower_case=True \
    eval_top_docs=[10] \
    train.warmup_steps=0 \
    train.num_train_epochs=3




