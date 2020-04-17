import torch
import itertools

def extract_arguments_crf(net,text, tokenizer, id2label):
    device = net.device
    tokens = tokenizer.tokenize(text)
    mapping = tokenizer.rematch(text,tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    token_ids = torch.tensor(token_ids, dtype=torch.long,device=device).unsqueeze(0)
    out = net(token_ids)
    labels = net.crf._viterbi_decode(out[0][0])[1]

    arguments, starting = [], False
    num_times = [(k, len(list(v))) for k, v in itertools.groupby(labels)]

    s = 0
    for i in num_times:
        if i[0]>0:
            arguments.append([list(range(s, s+i[1])),id2label[i[0]]])
        s += i[1]

    mapping[0] = mapping[1]
    mapping[-1] = mapping[-2]
    # print(arguments)
    # print(mapping)
    return {
        text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1]: l
        for w, l in arguments
    }

def extract_arguments_softmax(net,text, tokenizer, id2label):
    device = net.device
    tokens = tokenizer.tokenize(text)
    mapping = tokenizer.rematch(text,tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    token_ids = torch.tensor(token_ids, dtype=torch.long,device=device).unsqueeze(0)
    out = net(token_ids)
    labels = torch.max(out[0],2)[1]
    labels = labels.squeeze().tolist()
    arguments, starting = [], False
    num_times = [(k, len(list(v))) for k, v in itertools.groupby(labels)]
    s = 0
    for i in num_times:
        if i[0]>0:
            arguments.append([list(range(s, s+i[1])), id2label[i[0]]])
        s += i[1]

    mapping[0] = mapping[1]
    mapping[-1] = mapping[-2]
    # print(arguments)
    # print(mapping)
    return {
        text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1]: l
        for w, l in arguments
    }
