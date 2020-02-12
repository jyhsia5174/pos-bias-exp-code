import random as rd

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def random_select(fname, item_size):
    rd.seed(1)
    item_list = [i for i in range(item_size)]
    select_num = 10

    with open(fname, 'r') as rf, open("{}.rnd.select".format(fname), 'w') as of:

        for line in rf:
            label, features = line.strip().split(None, maxsplit=1)
            pos_label, dummy = label.split(":", maxsplit=1)

            select_item = rd.sample(item_list, select_num)
            try:
                idx = select_item.index( int(pos_label) )
            except:
                idx = -1

            select_item = [ "{}:0:1".format(i) for i in select_item ]

            if( idx != -1 ):
                select_item[idx] = label

            of.write("{} {}\n".format( ",".join(select_item), features ))


if __name__ == '__main__':
    item_size = file_len('item.ffm')
    random_select("filter.ffm", item_size)
