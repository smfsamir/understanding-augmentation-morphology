#!/bin/sh
SCRATCHDIR=$1
LANG=$2

mv "${SCRATCHDIR}/${LANG}_ids.pickle" "${SCRATCHDIR}/${LANG}_ids_true_initial.pickle"
mv "${SCRATCHDIR}/${LANG}_src_dict.pickle" "${SCRATCHDIR}/${LANG}_src_dict_true_initial.pickle"
mv "${SCRATCHDIR}/${LANG}_src_tokens.pickle" "${SCRATCHDIR}/${LANG}_src_tokens_true_initial.pickle"
mv "${SCRATCHDIR}/${LANG}_embeddings.pickle" "${SCRATCHDIR}/${LANG}_embeddings_true_initial.pickle"