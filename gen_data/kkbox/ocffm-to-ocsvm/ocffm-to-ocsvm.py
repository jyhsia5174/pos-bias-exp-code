import sys

def build_dict(item, context_list):
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

    for context in context_list:
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

def convert_feature( feature_list, convert_dict):
    converted_feature_list = []
    for feat in feature_list:
        converted_feature_list.append("{}:1".format(convert_dict[feat]))
    return converted_feature_list

def convert_item( item, item_dict):
    rf = open(item, 'r')
    of = open(item.replace("ffm", "svm"), 'w')

    for line in rf:
        tokens = line.strip().split()
        features = tokens
        converted_features = convert_feature(features, item_dict)
        of.write("{}\n".format(" ".join(converted_features)))

def convert_context( context, context_dict):
    rf = open(context, 'r')
    of = open(context.replace("ffm", "svm"), 'w')

    for line in rf:
        tokens = line.strip().split()
        label = tokens[0]
        features = tokens[1:]
        converted_features = convert_feature(features, context_dict)
        of.write("{} {}\n".format(label, " ".join(converted_features)))

if __name__ == '__main__':
    item = sys.argv[1]
    context_list = sys.argv[2:]
    print("item file: {} context files: [{}]".format(item, ','.join(context_list)))

    print('Start build dictionary')
    item_dict, context_dict = build_dict(item, context_list)

    convert_item( item, item_dict)
    for context in context_list:
        convert_context( context, context_dict)
