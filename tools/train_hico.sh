export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python -m torch.distributed.launch --nproc_per_node=8 --master_port 3992 --use_env main.py \
    --batch_size 32 \
    --output_dir ../checkpoints/hico_det/ \
    --epochs 100 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 64 \
    --enable_dec \
    --num_tokens 12 \
    --dataset_file hico  --set_cost_hoi_type 5\
    --enable_focal_loss --description_file_path ../buid_tree/hico-build-tree-embedding.json