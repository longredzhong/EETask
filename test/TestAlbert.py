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
# # %%
# from src.model.AlbertSoftMax import AlbertSoftmaxForNer
# AlbertSoftmaxForNer = AlbertSoftmaxForNer.from_pretrained(
#     pretrained_model_name_or_path=albert_config.pretrained_path,
#     config=albert_config)
# # %% 测试训练
# train_loader = EE.get_train_data_loader()
# from torch.optim import Adam
# AlbertSoftmaxForNer = AlbertSoftmaxForNer.to(device)
# optim = Adam(AlbertSoftmaxForNer.parameters(), lr=1e-5)
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
# %%
from src.model.AlbertCRF import AlbertCrfForNer
AlbertCrfForNer = AlbertCrfForNer.from_pretrained(
    pretrained_model_name_or_path=albert_config.pretrained_path,
    config=albert_config).to(device)
#%%
train_loader = EE.get_train_data_loader()
from torch.optim import Adam
optim = Adam([{'params': AlbertCrfForNer.bert.parameters()},
              {'params': AlbertCrfForNer.crf.parameters(), 'lr': 100}
            ], lr=1e-5)
# %%
# from torch.nn import DataParallel
# AlbertCrfForNer = DataParallel(AlbertCrfForNer,device_ids=[1,0])
# %%
# %%
for i in train_loader:
    AlbertCrfForNer.train()
    optim.zero_grad()
    loss, out = AlbertCrfForNer(input_ids=i.input_ids.to(device),
                                attention_mask=i.attention_mask.to(device),
                                token_type_ids=i.token_type_ids.to(device),
                                position_ids=None,
                                head_mask=None,
                                labels=i.label_ids.to(device),
                                input_lens = i.seq_len)
    loss.backward()
    optim.step()
    print(loss.item())
# # %%
# AlbertCrfForNer.save_pretrained("..")
#%%
AlbertCrfForNer = AlbertCrfForNer.cpu()
# %%
AlbertCrfForNer.crf._viterbi_decode(out[1])[1]

# %%
import src.util.extract_arguments as extract_arguments
a = extract_arguments.extract_arguments_crf(
    AlbertCrfForNer, "通用汽车泰国裁员350人，占员工总数的15%左右", EE.tokenizer, EE.id2label)

a

# %%
from src.dataloader.utils import load_data
data = load_data("/home/longred/EETask/data/dev.json")
# %%
for text,au in data:
    print(extract_arguments.extract_arguments_crf(
        AlbertCrfForNer, text, EE.tokenizer, EE.id2label)
    )
