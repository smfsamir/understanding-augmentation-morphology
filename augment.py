import pdb
import align
import argparse
import codecs
import os, sys
from random import random, choice
import re
from typing import List, Tuple
from packages.utils.constants import HALL_DATA_PATH


def read_data(filename):
    with codecs.open(filename, 'r', 'utf-8') as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for l in lines:
        l = l.strip().split('\t')
        if l:
            inputs.append(list(l[0].strip()))
            outputs.append(list(l[1].strip()))
            tags.append(re.split('\W+', l[2].strip()))
    return inputs, outputs, tags

def find_good_range(a: str,b: str):
    """Finds ranges that align.
    Makes sure there are non-aligned strings, I think...

    Args:
        a (str): Source lemma.
        b (str): Target inflected form.

    Returns:
        [List[Tuple[int]]]: List of ranges (two-tuples). 
    """
    mask = [(a[i]==b[i] and a[i] != u" ") for i in range(len(a))] # mask of aligned characters.
    if sum(mask) == 0:
        # Some times the alignment is off-by-one
        b = ' ' + b
        mask = [(a[i]==b[i] and a[i] != u" ") for i in range(len(a))]
    ranges = []
    prev = False # previous match
    for i,k in enumerate(mask): # current match
        if k and prev: # if current match and previous match
            prev = True
        elif k and not prev: # elif current match but not prev match
            start = i
            prev = True
        elif prev and not k: # prev but not current
            end = i
            ranges.append((start, end))
            prev = False
        elif not prev and not k:
            prev = False
    if prev: # rest of the string 
        ranges.append((start,i+1)) 
    ranges = [c for c in ranges if c[1]-c[0]>2] # must be contiguous strings greater than length 2
    return ranges

def obtain_invariant_indices(output_aligned: str, good_indices_range: List[Tuple[int]]):
    """Obtain the indices that haven't changed in the augmented string.

    Args:
        output_aligned (str): Output form, possibly with spaces (for alignment)
        good_indices_range (List[Tuple[int]]): Overlapping indices found after alignment
    """
    gt_str = re.sub(r"\s", "", output_aligned)
    i = 0
    j = 0
    # aligned_to_gt_inds = {} 
    gt_to_aligned_inds = {} 
    while i < len(gt_str):
        while output_aligned[j] != gt_str[i]:
            j += 1
        gt_to_aligned_inds[i] = j
        i += 1
        j += 1
    invariant_inds = []
    varying_inds = []
    for ind in sorted(gt_to_aligned_inds.keys()):
        aligned_i = gt_to_aligned_inds[ind]
        for s_e_range in good_indices_range:
            if s_e_range[0] <= aligned_i and aligned_i < s_e_range[1]:
                varying_inds.append(ind)
    for i in range(len(gt_str)):
        if i not in varying_inds:
            invariant_inds.append(i)
    assert not invariant_inds == [] or (len(good_indices_range)== 1 and good_indices_range[0][1] == len(gt_str)), f"{output_aligned},{good_indices_range}" 
    return invariant_inds

def augment(inputs, outputs, tags, characters):
    temp = [(''.join(inputs[i]), ''.join(outputs[i])) for i in range(len(outputs))]
    aligned = align.Aligner(temp).alignedpairs

    vocab = list(characters)
    try:
        vocab.remove(u" ")
    except:
        pass

    new_inputs = []
    new_outputs = []
    new_tags = []
    invariant_target_inds = [] # going to be a list of lists
    source_indices = []

    for k,item in enumerate(aligned):
        i,o = item[0],item[1]
        good_range = find_good_range(i,o)
        invariant_inds = obtain_invariant_indices(o, good_range)
        if good_range and invariant_inds: # NOTE: i added the comment
            new_i, new_o = list(i), list(o)
            for r in good_range:
                s = r[0]
                e = r[1]
                # if (e-s>5): #arbitrary value. NOTE: does this matter..?
                #     s += 1
                #     e -= 1
                for j in range(s,e):
                    if random() > 0.5: #arbitrary value
                        nc = choice(vocab)
                        new_i[j] = nc
                        new_o[j] = nc
            new_i1 = [c for l,c in enumerate(new_i) if (c.strip() or (new_i[l]==' ' and new_o[l] == ' '))]
            new_o1 = [c for l,c in enumerate(new_o) if (c.strip() or (new_i[l]==' ' and new_o[l] == ' '))]
            new_inputs.append(new_i1)
            new_outputs.append(new_o1)
            new_tags.append(tags[k])
            invariant_target_inds.append(invariant_inds)
            # assert all([("".join(o.split(" "))[x]==new_o1[x]) for x in invariant_inds]), (f"{o} and {new_o1} and {invariant_inds}")
            # assert len(new_i1) == len(i)
            # assert len(new_o1) == len(o)
            source_indices.append(k)
        else:
            new_inputs.append([])
            new_outputs.append([])
            new_tags.append([])
            invariant_target_inds.append([])
            source_indices.append(False)
    assert len(new_inputs) == len(invariant_target_inds)
    return new_inputs, new_outputs, new_tags, invariant_target_inds, source_indices

def get_chars(l):
    flat_list = [char for word in l for char in word]
    return list(set(flat_list))


parser = argparse.ArgumentParser()
parser.add_argument("datapath", help="path to data", type=str)
parser.add_argument("language", help="language", type=str)
parser.add_argument("--examples", help="number of hallucinated examples to create (def: 10000)", default=10000, type=int)
parser.add_argument("--use_dev", help="whether to use the development set (def: False)", action="store_true")
args = parser.parse_args()

DATA_PATH = args.datapath
L2 = args.language
LOW_PATH = os.path.join(DATA_PATH, L2)

N = args.examples
usedev = args.use_dev

lowi, lowo, lowt = read_data(LOW_PATH)

vocab = get_chars(lowi+lowo)

i,o,t, all_invariant_o_inds, all_source_indices = [], [], [], [], []
while len(i) < N:
    if usedev:
        # Do augmentation also using examples from dev
        ii,oo,tt, invariant_o_inds, curr_source_indices = augment(lowi, lowo, lowt, vocab)
    else:
        # Just augment the training set
        ii,oo,tt, invariant_o_inds, curr_source_indices = augment(lowi, lowo, lowt, vocab)
    ii = [c for c in ii if c]
    oo = [c for c in oo if c]
    tt = [c for c in tt if c]
    invariant_o_inds = [c for c in invariant_o_inds if c]
    i += ii
    o += oo
    t += tt
    all_invariant_o_inds += invariant_o_inds
    all_source_indices += curr_source_indices
    if len(ii) == 0:
        break
# Wait is this needed?
i = [c for c in i if c]
o = [c for c in o if c]
t = [c for c in t if c]
all_invariant_o_inds = [c for c in all_invariant_o_inds if c]
all_source_indices = [c for c in all_source_indices if c]

with codecs.open(os.path.join(HALL_DATA_PATH,L2+"-hall" + f"-{str(N)}"), 'w', 'utf-8') as outp:
    for k in range(min(N, len(i))):
        outp.write(''.join(i[k]) + '\t' + ''.join(o[k]) + '\t' + ';'.join(t[k]) + '\t' + str(all_invariant_o_inds[k]) + '\t' + str(all_source_indices[k]) + '\n')