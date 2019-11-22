import sys

def build_dict(item, context_lst):
    counter = 1
    item_dict = {}
    context_dict = {}

    rf = open(item, 'r')
    for line in rf:
        feats = line.strip().split()
        for tk in feats:
            item_dict.setdefault(tk, -1)
            if item_dict[tk] == -1:
                item_dict[tk] = counter
                counter += 1
    rf.close()

    for context in context_lst:
        rf = open(context, 'r')
        for line in rf:
            toks = line.strip().split()
            feats = toks[1:]
            for tk in feats:
                context_dict.setdefault(tk, -1)
                if context_dict[tk] == -1:
                    context_dict[tk] = counter
                    counter += 1

    return (item_dict, context_dict)

def build_items_feat(item, item_dict):
    items_to_feat = {}
    rf = open(item, 'r')
    for i, line in enumerate(rf):
        items_to_feat[i] = map(lambda x: item_dict[x], line.strip().split())
    rf.close()
    return items_to_feat

def format_output(of, item_feat, context_feat, click):
    all_feat = item_feat + context_feat
    formated_feat = map(lambda x: "{}:1".format(x), sorted(all_feat))
    of.write("{} {}\n".format( click, " ".join(formated_feat)))

def convert( item, context, item_dict, context_dict):
    rf = open(context, 'r')
    of = open(context.replace('ffm', 'svm'), 'w')

    items_to_feat = build_items_feat(item, item_dict)
    for line in rf:
        toks = line.strip().split()
        labels = map( lambda x: x.split(':'), toks[0].strip(',').split(','))
        feats = toks[1:]
        context_feat = map(lambda x: context_dict[x], feats)
        for item_id, click in labels:
            item_feat = items_to_feat[int(item_id)]
            format_output(of, item_feat, context_feat, click)

    rf.close()
    of.close()

def output_map( item, context):
    rf = open(context, 'r')
    of = open(context.replace('ffm', 'map'), 'w')

    for line in rf:
        toks = line.strip().split()
        labels = map( lambda x: x.split(':'), toks[0].strip(',').split(','))
        for item_id, click in labels:
            of.write("{} {}\n".format(click, item_id))

    rf.close()
    of.close()

if __name__ == '__main__':
    item = sys.argv[1]
    context_lst = sys.argv[2:]
    print("item file: {} context files: [{}]".format(item, ','.join(context_lst)))

    print('Start build dictionary')
    item_dict, context_dict = build_dict(item, context_lst)
    print('Start converting')
    for context in context_lst:
        print("Convert {}".format( context ) )
        convert(item, context, item_dict, context_dict)
        #output_map(item, context)
