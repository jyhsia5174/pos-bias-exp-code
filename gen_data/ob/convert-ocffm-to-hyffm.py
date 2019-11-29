import sys

def convert(line):
    toks = line.strip().split()
    labels = toks[0].strip(',').split(',')
    features = toks[1:]

    converted_label = [] 
    for label in labels:
        converted_label.append("{}:1".format(label))

    return "{} {}".format(",".join(converted_label), " ".join(features))

if __name__ == '__main__':
    print( sys.argv[1])
    f1 = open(sys.argv[1], 'r')
    f2 = open(sys.argv[1].replace('ffm', 'hyffm'), 'w')

    for line in f1:
        converted_line = convert(line)
        f2.write("{}\n".format(converted_line))

    f1.close()
    f2.close()

