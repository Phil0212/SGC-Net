python -m torch.distributed.launch --nproc_per_node=1 --master_port 3093 --use_env main.py \
    --batch_size 32 \
    --output_dir ../checkpoints/hico_det/ \
    --epochs 100 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 64 \
    --num_tokens 12 \
    --enable_dec \
    --dataset_file hico \
    --enable_focal_loss --description_file_path ../buid_tree/hico-build-tree-embedding.json \
    --eval --pretrained ../checkpoints/hico_det/model.pth \

