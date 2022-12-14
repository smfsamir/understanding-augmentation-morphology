from sys import stderr
import click
from collections import defaultdict as dd
from math import log
from random import random, choice

from data import readdata, Datum

def entropy(d):
    vals = [1.0 * v for v in d.values() if v > 0]
    vals = [v/sum(vals) for v in vals]
    return -sum([x*y for x,y in zip(vals,[log(v) for v in vals])])

def getform(form,stem,ranges):
    replacements = {}
    for part in stem:
        replacements[part] = (stem[part][:ranges[part][0]],part,stem[part][ranges[part][1]:])
    newform = [replacements[x] if x in replacements else [x] for x in form]
    newform = [x for y in newform for x in y]
    collapsed = ['']
    for x in newform:
        if x in replacements:
            if ranges[x][1] - ranges[x][0] > 0:
                collapsed.append(x)
                collapsed.append('')
        else:
            collapsed[-1] += x
    collapsed = tuple([c for c in collapsed if c!= ''])
    return collapsed

def sample(d, formcounts):
    e0 = sum([entropy(d) for d in formcounts.values()])
    part = choice(list(d[1][0].keys()))
    dir = choice([-1,1])
    edge = choice([0,1])
    if edge == 0:
        if d[2][0][part][0] + dir < 0 or d[2][0][part][0] + dir > d[2][0][part][1]:
            return e0
    else:
        if d[2][0][part][1] + dir > len(d[1][0][part]) or d[2][0][part][1] + dir < d[2][0][part][0]:
            return e0
    for l in d[0]:
        if not d[0][l]:
            continue
        formcounts[l][getform(d[0][l],d[1][0],d[2][0])] -= 1
    d[2][0][part][edge] += dir
    for l in d[0]:
        if not d[0][l]:
            continue
        formcounts[l][getform(d[0][l],d[1][0],d[2][0])] += 1
    e1 = sum([entropy(d) for d in formcounts.values()])

    if e0 < e1:
        for l in d[0]:
            if not d[0][l]:
                continue
            formcounts[l][getform(d[0][l],d[1][0],d[2][0])] -= 1
        d[2][0][part][edge] -= dir
        for l in d[0]:
            if not d[0][l]:
                continue
            formcounts[l][getform(d[0][l],d[1][0],d[2][0])] += 1
        assert(sum([entropy(d) for d in formcounts.values()]) == e0)
        return e0
    return e1

def add(d,formdict,dir=1):
    for l in formdict:
        form = d.getform(l)
        if form:
            formdict[l][d.getform(l)] += dir

def getprob(d,formdict):
    prob = 1.0
    for l in formdict:
        tot = 1.0 * sum(formdict[l].values())
        form = d.getform(l)
        if form:
            assert(formdict[l][form] > 0)
            prob *= formdict[l][form] / tot
    return prob

def entropy(formdict):
    h = 0
    for l in formdict:
        tot = 1.0 * sum(formdict[l].values())
        ps = [p/tot for p in formdict[l].values() if p != 0]
        h += -sum(log(p)*p for p in ps)
    return h

def loglikelihood(formdict):
    ll = 0
    for l in formdict:
        tot = 1.0 * sum(formdict[l].values())
        ll += sum([log(x/tot) for x in formdict[l].values() if x > 0])
    return ll

@click.command()
@click.option("--data_file", required=True)
@click.option("--epochs", required=True)
@click.option("--res_file", required=True)
def main(data_file, epochs, res_file):
    epochs = int(epochs)
    print("Read paradigms",file=stderr)

    data, labels = readdata(data_file)
    
    formcounts = {l:dd(lambda : 0) for l in labels}

    for d in data:
        for l in d[0]:
            form = d[0][l]
            if form:
                formcounts[l][form] += 1
        d.append([{x:[0,len(y)] for x,y in stem.items()} for stem in d[1]])

    data = [Datum(d) for d in data]
    formdict = {l:dd(lambda : 0) for l in labels}
    for d in data:
        for l in labels:
            form = d.getform(l)
            if form:
                formdict[l][form] += 1

    print(f"Train for {epochs} epochs",file=stderr)
    for i in range(epochs):
        for d in data:
            prob0 = getprob(d,formdict)
            add(d,formdict,-1)
            config = d.sampleconfig()
            success = d.apply(config)
            if success:
                add(d,formdict)
                prob1 = getprob(d,formdict)
                if prob1 / prob0 > 1:
                    pass
                else:
                    add(d,formdict,-1)
                    d.apply(config,invert=1)
                    add(d,formdict)
            else:
                add(d,formdict)
        stderr.write(f"Epoch {i+1}: LL: %f, Entropy: %f\n" % (loglikelihood(formdict),entropy(formdict)))

    res_file = open(res_file, "w")
    for d in data:
        forms = '#'.join(['+'.join(d.getform(l))+':'+l for l in labels if d.getform(l)])
        stem = d.getstem()
        print('%s\t%s' % (forms,stem), file=res_file)

if __name__=="__main__":
    main()
