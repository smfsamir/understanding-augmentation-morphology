#!/bin/sh

SCRATCHDIR=$1
LANG=$2
ALGORITHM=$3

cd "$SCRATCHDIR" || exit
PREPROCESS="${LANG}_fairseq_bin"
SAVEPREF="${LANG}_${ALGORITHM}_model_checkpoints"

fairseq-generate $PREPROCESS \
    --path $SAVEPREF/checkpoint_best.pt \
    --batch-size 128 --beam 5 --required-batch-size-multiple 4 > "${SCRATCHDIR}/${LANG}_results.txt"

# fairseq-interactive --path $SAVEPREF/checkpoint_best.pt $PREPROCESS --source-lang src --target-lang tgt --buffer-size 100 < ${SCRATCHDIR}/${LANG}-test.src   > "${SCRATCHDIR}/${LANG}_results.txt"