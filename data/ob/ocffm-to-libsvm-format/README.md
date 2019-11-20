# Convert ocffm format to libsvm format

## Remap index
Input: item.ffm tr.ffm va.ffm gt.ffm
Output: tr.svm va.svm gt.svm

- Config convert.sh
```shell
./convert.sh
```

## Built trva.svm
```shell
cat va.svm tr.svm > trva.svm
```

