from sys import stderr

from random import choice

class Datum:
    def __init__(self,d):
        self.formdict = d[0]
        self.stem = d[1][0]
        self.ranges = {sp:[0,len(self.stem[sp])] for sp in self.stem}

    def getform(self,label):
        form = self.formdict[label]
        if form == None:
            return form
        replacements = {sp:(self.stem[sp][:self.ranges[sp][0]],
                            sp,
                            self.stem[sp][self.ranges[sp][1]:]) for sp in self.stem}
        form = [replacements[x] if x in self.stem else [x] for x in form]
        form = [x for y in form for x in y]
        newform = ['']
        for x in form:
            if x.isdigit():
                if self.stem[x][self.ranges[x][0]:self.ranges[x][1]] != '':
                    newform.append(x)
                    newform.append('')
            else:
                newform[-1] += x
        return tuple([x for x in newform if x != ''])

    def getstem(self):
        return ','.join(['%s=%s' % (k,v[self.ranges[k][0]:
                                        self.ranges[k][1]]) 
                         for k,v in self.stem.items()])

    def sampleconfig(self):
        return (choice(list(self.stem)), choice([0,1]),choice([-1,1]))

    def apply(self,config,invert=0):
        sp = config[0]
        edge = config[1]
        dir = -config[2] if invert else config[2]
        if dir > 0:
            if edge == 0:
                if self.ranges[sp][edge] >= self.ranges[sp][1]:
                    return 0
            else:
                if self.ranges[sp][edge] >= len(self.stem[sp]):
                    return 0
        else:
            if edge == 0:
                if self.ranges[sp][edge] == 0:
                    return 0
            else:
                if self.ranges[sp][edge] <= self.ranges[sp][0]:
                    return 0
        self.ranges[sp][edge] += dir
        return 1

def getlabeldict(forms,labels):
    d = {label:None for label in labels}
    for form, label in forms:
        d[label] = tuple(form.split('+'))
    return d

def getstem(s):
    parts = [x.split('=') for x in s.split(',')]
    return {var: value for var,value in parts}

def readdata(fn):
    data = []
    labels = set()
    for line in open(fn):
        line = line.strip()
        fields = line.split('\t')
        forms = [x.split(':') for x in fields[0].split('#')]
        labels.update([label for _, label in forms])
        stems = fields[1].split('#')
        data.append((forms,stems))
    data = [[getlabeldict(d[0],labels),
             [getstem(s) for s in d[1]]] for d in data]
    return data, list(labels)

def encode(data,labels):
    encoded = []
    mask = []
    encoder = {}
    for forms,stems in data:
        line = []
        maskline = []
        for l in labels:
            if not forms[l]:
                line.append(float('nan'))
                maskline.append(0)
            else:
                id = 0
                if not forms[l] in encoder:
                    encoder[forms[l]] = len(encoder) + 1
                line.append(encoder[forms[l]])
                maskline.append(1)
            encoded.append(line)
            mask.append(maskline)
    return encoded, mask, {id:form for form, id in encoder.items()} 

def interleave(form, stem):
    form = [(stem[x] if x in stem else x) for x in form]
    return ''.join(form)
