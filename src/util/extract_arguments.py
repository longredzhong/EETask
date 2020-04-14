import torch

def extract_arguments_crf(net,text, tokenizer, id2label):
    device = net.device
    tokens = tokenizer.tokenize(text)
    mapping = tokenizer.rematch(text,tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    token_ids = torch.tensor(token_ids, dtype=torch.long,device=device).unsqueeze(0)
    out = net(token_ids)
    labels = net.crf._viterbi_decode(out[0][0])[1]
    arguments = []
    for i, label in enumerate(labels):
        if label not in (0,218,219):#TODO 硬编码  等会改
            arguments.append([[i], id2label[(label)]])
    mapping[0] = mapping[1]
    mapping[-1] = mapping[-2]
    print(arguments)
    print(mapping)
    return {
        text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1]: l
        for w, l in arguments
    }
