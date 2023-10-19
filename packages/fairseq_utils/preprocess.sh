#!/bin/sh

## TODO: cd into scratch directory?
SCRATCHDIR=$1
LANG=$2
TEST_SPLIT_TYPE=$3

echo $SCRATCHDIR
echo $LANG
cd "$SCRATCHDIR" || exit
DESTDIR="${LANG}_fairseq_bin"
mkdir -p "${DESTDIR}"
fairseq-preprocess --source-lang src --target-lang tgt --trainpref $LANG-train --validpref $LANG-valid --testpref "${LANG}-test"  --destdir $DESTDIR 