## 0. Data format

Input data consists of (partial) morphological paradigms in the following 2-column format:

```
Aadamiksi	case=translative,number=singular
Aadamein	case=instructive,number=plural

Aamuineen	case=comitative,number=plural
Aamut	case=nominative,number=plural

Aatoihin	case=illative,number=plural
Aatona	case=essive,number=singular
```

## 1. Extract paradigms

```python3 src/pextract.py --data_file data/fitrain.2 --res_file data/fitrain.2.par```

## 2. Resample stems

```python3 src/stemsample.py --data_file data/fitrain.2.par --res_file data/fitrain.2.par.resample --epochs 50```