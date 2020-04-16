#%%

import os
import sys
sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
#%%
from transformers.configuration_albert import AlbertConfig
albert_config = AlbertConfig.from_pretrained(
    r"/home/longred/EETask/prev_trained_model/albert_large_zh")
albert_config.pretrained_path = r"/home/longred/EETask/prev_trained_model/albert_large_zh/"
albert_config.vocab_path = r"/home/longred/EETask/prev_trained_model/albert_large_zh/vocab.txt"
albert_config.train_data_path = r"/home/longred/EETask/data/train.json"
albert_config.batch_size = 12
albert_config.event_schema_path = r"/home/longred/EETask/data/event_schema.json"
albert_config.pretrained_path = r"/home/longred/EETask/prev_trained_model/albert_large_zh"
#%%
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from src.dataloader.Dataset import EETaskDataloader
EE = EETaskDataloader(albert_config)
train_loader = EE.get_train_data_loader()
albert_config.num_labels = EE.num_labels
albert_config.label2id = EE.label2id
albert_config.id2label = EE.id2label
# %%
# from src.model.AlbertSoftMax import AlbertSoftmaxForNer
# AlbertSoftmaxForNer = AlbertSoftmaxForNer.from_pretrained(
#     pretrained_model_name_or_path=albert_config.pretrained_path,
#     config=albert_config)
# # %% 测试训练
# train_loader = EE.get_train_data_loader()
# from torch.optim import Adam
# AlbertSoftmaxForNer = AlbertSoftmaxForNer.to(device)
# optim = Adam(AlbertSoftmaxForNer.parameters(), lr=0.01)
# #%%
# for i in train_loader:
#     AlbertSoftmaxForNer.train()
#     optim.zero_grad()
#     loss, out = AlbertSoftmaxForNer(input_ids = i.input_ids.to(device),
#                                     attention_mask=i.attention_mask.to(device),
#                                     token_type_ids=i.token_type_ids.to(device),
#                                     position_ids=None,
#                                     head_mask=None,
#                                     labels=i.label_ids.to(device))
#     loss.backward()
#     optim.step()
#     print(loss.item())

# # %%
# import src.util.extract_arguments as e
# e.extract_arguments_softmax(AlbertSoftmaxForNer,"撒范德萨发撒法撒旦飞洒",EE.tokenizer,EE.id2label)
# #%%
# out = AlbertSoftmaxForNer(i.input_ids[0].unsqueeze(0).cuda())
# #%%
# labels = torch.max(out[0], 2)[1]
# #%%
# labels.squeeze().tolist()
# #%%
# i.label_ids[0]
# #%%
# from src.dataloader.utils import load_data
# data = load_data("/home/longred/EETask/data/dev.json")
# # %%
# for text, au in data:
#     print(e.extract_arguments_softmax(
#         AlbertSoftmaxForNer, text, EE.tokenizer, EE.id2label)
#     )
# # %%
# while (loss.item() >2 and loss.item()<7):
#     for i in train_loader:
#         AlbertSoftmaxForNer.train()
#         optim.zero_grad()
#         loss, out = AlbertSoftmaxForNer(input_ids=i.input_ids.to(device),
#                                         attention_mask=i.attention_mask.to(device),
#                                         token_type_ids=i.token_type_ids.to(device),
#                                         position_ids=None,
#                                         head_mask=None,
#                                         labels=i.label_ids.to(device))
#         loss.backward()
#         optim.step()
#         print(loss.item())
#%%
from src.model.AlbertCRF import AlbertCrfForNer
AlbertCrfForNer = AlbertCrfForNer.from_pretrained(
    pretrained_model_name_or_path=albert_config.pretrained_path,
    config=albert_config).to(device)
#%%
train_loader = EE.get_train_data_loader()
#%%
from torch.optim import Adam
optim = Adam(
    [{'params': AlbertCrfForNer.crf.parameters(), 'lr': 0.1}], lr=0.0001)

#%%
# from torch.nn import DataParallel
# AlbertCrfForNer = DataParallel(AlbertCrfForNer,device_ids=[1,0])
#%%

from tqdm import tqdm
tt = tqdm(train_loader)
for i in tt:
    AlbertCrfForNer.train()
    loss, out = AlbertCrfForNer(input_ids=i.input_ids.to(device),
                                attention_mask=i.attention_mask.to(device),
                                token_type_ids=i.token_type_ids.to(device),
                                position_ids=None,
                                head_mask=None,
                                labels=i.label_ids.to(device),
                                input_lens = i.seq_len)
    loss.backward()
    optim.step()
    optim.zero_grad()
    tt.set_postfix(loss=loss.item())
    # print(loss.item())
#%%
def train(net, train_loader, optim):
    loader = tqdm(train_loader)
    for i in loader:
        AlbertCrfForNer.train()
        loss, out = AlbertCrfForNer(input_ids=i.input_ids.to(device),
                                attention_mask=i.attention_mask.to(device),
                                token_type_ids=i.token_type_ids.to(device),
                                position_ids=None,
                                head_mask=None,
                                labels=i.label_ids.to(device),
                                input_lens=i.seq_len)
        loss.backward()
        optim.step()
        optim.zero_grad()
        loader.set_postfix(loss=loss.item())
#%%
from src.dataloader.utils import load_data
from src.util.utils import lcs
data = load_data("/home/longred/EETask/data/dev.json")
import src.util.extract_arguments as extract_arguments
def evaluate(data):
    AlbertCrfForNer.eval()
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for text, arguments in tqdm(data):
        inv_arguments = {v: k for k, v in arguments.items()}
        pred_arguments = extract_arguments.extract_arguments_crf(
            AlbertCrfForNer, text, EE.tokenizer, EE.id2label)
        pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
        Y += len(pred_inv_arguments)
        Z += len(inv_arguments)
        for k, v in pred_inv_arguments.items():
            if k in inv_arguments:
                # 用最长公共子串作为匹配程度度量
                l = lcs(v, inv_arguments[k])
                X += 2. * l / (len(v) + len(inv_arguments[k]))
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

#%%
with torch.no_grad():
    a = evaluate(data)
    print(a)
#%%
train(AlbertCrfForNer,train_loader,optim)
#%%
for i in range(2):
    train(AlbertCrfForNer, train_loader, optim)
    with torch.no_grad():
        a = evaluate(data)
        print(a)
#%%
# AlbertCrfForNer.save_pretrained("..")
#%%
# AlbertCrfForNer = AlbertCrfForNer.cpu()
# %%
# AlbertCrfForNer.crf._viterbi_decode(out[1])[1]

# %%

a = extract_arguments.extract_arguments_crf(
    AlbertCrfForNer, "通用汽车泰国裁员350人，占员工总数的15%左右", EE.tokenizer, EE.id2label)

a

# %%

# %%
for text,au in data:
    print(extract_arguments.extract_arguments_crf(
        AlbertCrfForNer, text, EE.tokenizer, EE.id2label)
    )



# %%


# %%
