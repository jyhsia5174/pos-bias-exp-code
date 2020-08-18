
def read_rank_list():
    rf = open('popular_rank', 'r')
    rank_list = []
    for line in rf:
        label = line.strip()
        rank_list.append(label)
    rf.close()

    return rank_list

def get_new_label( rank_list, clicks):
    res = []
    for i in range(len(rank_list)):
        res.append("{}:{}:1".format(rank_list[i], clicks[i]))

    return ','.join(res)

def filter_by_popular(rank_list):
    rf = open('filter.ffm', 'r')
    of = open('popular.ffm', 'w')

    for line in rf:
        label, feats = line.strip().split(None, 1)

        label_rank = rank_list.index(label.split(':')[0])
        clicks = [0]*10
        clicks[label_rank] = 1

        new_label = get_new_label( rank_list, clicks )

        of.write("{} {}\n".format( new_label, feats ) )

    of.close()
    rf.close()

rank_list = read_rank_list()
filter_by_popular(rank_list)
