from sys import argv

import numpy as np
from fancyimpute import soft_impute, nuclear_norm_minimization

from data import readdata, encode, interleave

data, labels = readdata(argv[1])
encoded,mask,decoder = encode(data,labels)
imputer = soft_impute.SoftImpute(max_iters=100)
imputed = imputer.complete(np.array(encoded))#,np.array(mask))

for i, d in enumerate(data):
    line = imputed[i]
    for l, val in zip(labels,line):
        if d[0][l] == None:
            val = int(round(val))
            if val <= 0:
                val = 1
            if val >= len(decoder):
                val = len(decoder)
            d[0][l] = decoder[int(round(val))]

outputf = open("%s.imputed" % argv[1],"w")
for d in data:
    for stem in d[1]:
        for l in labels:
            outputf.write("%s\t%s\n" % (interleave(d[0][l],stem),l))
        outputf.write('\n')
