import os, sys

root = sys.argv[1]
mode = sys.argv[2]

f_path = os.path.join(root, "%s.record"%mode)
va_num = [2] 
te_num = list(range(4, 25, 4)) + list(range(5, 26, 4))
with open(f_path, 'r') as f:
    for i, line in enumerate(f):
        if i+1 in va_num:
            print(line.strip().split(' ')[0])
            print(line.strip().split(' ')[1])
        if i+1 in te_num:
            print(line.strip().split(' ')[-1])


