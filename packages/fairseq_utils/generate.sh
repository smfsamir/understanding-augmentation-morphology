#!/bin/sh

SCRATCHDIR=$1
LANG=$2

cd "$SCRATCHDIR" || exit
PREPROCESS="${LANG}_fairseq_bin"
SAVEPREF="${LANG}_model_checkpoints"


fairseq-generate $PREPROCESS \
    --path $SAVEPREF/checkpoint_best.pt \
    --batch-size 128 --beam 5 > "${SCRATCHDIR}/${LANG}_results.txt"