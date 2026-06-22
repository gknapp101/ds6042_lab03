# These commands for setting up/training/fine tuning the depth of 8 model
export DEPTH=8
export DEVICE_BATCH_SIZE=8         # MIG slice has limited VRAM; lower further if you OOM
export NCHAT_DATA=/scratch/$USER/lab03/data
mkdir -p "$NCHAT_DATA"

bash runs/speedrun.sh --download-only
python -m scripts.tok_train --vocab-size 8192
torchrun --nproc_per_node=1 -m scripts.base_train \
    --depth $DEPTH --device-batch-size $DEVICE_BATCH_SIZE \
    --num-iterations 2000 2>&1 | tee train.log

# fine tuning    
python -m scripts.chat_sft --num-iterations 800
python -m scripts.chat_cli
python -m scripts.chat_cli --source base # baseline model comparsion 
