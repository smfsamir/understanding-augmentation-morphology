#!/bin/sh

SCRATCHDIR=$1
LANG=$2
ALGORITHM=$3
SEED=$4

cd "$SCRATCHDIR" || exit
PREPROCESS="${LANG}_fairseq_bin"
SAVEPREF="${LANG}_${ALGORITHM}_model_checkpoints"
mkdir -p "${SAVEPREF}"

fairseq-train $PREPROCESS \
    --no-epoch-checkpoints \
    --source-lang src \
    --target-lang tgt \
    --save-dir $SAVEPREF \
    --seed $SEED \
    --arch transformer \
    --encoder-layers 4 \
    --decoder-layers 4 \
    --encoder-embed-dim 256 \
    --decoder-embed-dim 256 \
    --encoder-ffn-embed-dim 512 \
    --decoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --dropout 0.3 \
    --attention-dropout 0.1 \
    --relu-dropout 0 \
    --weight-decay 0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-betas '(0.9, 0.999)' \
    --batch-size 16 \
    --clip-norm 0 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-7 --lr 0.001 --stop-min-lr 1e-9 \
    --keep-interval-updates 20 \
    --max-tokens 2000 \
    --max-update 6000 \
    --update-freq 1 \
    --wandb-project subset-selection-augmentation \
    --log-format json --log-interval 20  