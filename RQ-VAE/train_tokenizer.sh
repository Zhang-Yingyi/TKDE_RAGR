python ./RQ-VAE/main.py \
  --device cuda:0 \
  --data_path ../data/Beauty/Beauty.emb-t5-td.npy\
  --alpha 0.01 \
  --beta 0.0001 \
  --cf_emb ./RQ-VAE/ckpt/Beauty-32d-sasrec.pt\
  --ckpt_dir ../checkpoint/