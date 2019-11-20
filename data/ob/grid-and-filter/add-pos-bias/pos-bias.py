import sys
import random as rd

file=sys.argv[1]

def init_pos_bias( gamma ):
    res_list = []
    base_rate = 0.9
    alpha_k = 0.9
    for i in range(20):
        res_list.append( gamma * alpha_k )
        alpha_k *= base_rate
    return res_list

def change_label_to_zero( tk ):
    idx, label, prop = tk.split(":")
    return "{}:{}:{}".format(idx, '0', prop)

def output_bias_file( file_name ):
    ofile_name = "{}.pos.bias".format(file_name)
    rf = open(file_name, 'r')
    of = open(ofile_name, 'w')
    pos_bias_list = init_pos_bias( 1.0 )
    print(pos_bias_list)
    for line in rf:
        toks = line.strip().strip(',').split(',')
        len(toks)
        for i, tk in enumerate(toks):
            if rd.random() >= pos_bias_list[i]:
                toks[i] = change_label_to_zero(tk)
        of.write("{}\n".format(','.join(toks)))

if __name__ == '__main__':
    output_bias_file(file)
