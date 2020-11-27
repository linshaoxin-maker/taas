# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
python finetune.py \
    --learning_rate=3e-5 \
    --fp16 \
    --gpus 2 \
    --do_train \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --freeze_encoder \
    --freeze_embeds \
    --label_smoothing 0.1 \
    $@
