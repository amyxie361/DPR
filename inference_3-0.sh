for id in 0
do
rm -rf /home/y247xie/01_exps/DPR/index/exp_test12348/
python generate_dense_embeddings.py \
    model_file=/home/y247xie/01_exps/DPR/ckpts/nq_12348/dpr_biencoder.34.920 \
    ctx_src=dpr_wiki \
    shard_id=${id} \
    num_shards=8 \
    out_file=/home/y247xie/01_exps/DPR/index/exp_test12348/index_${id} \
    batch_size=64
done


