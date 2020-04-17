#%%
import os
import sys
sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
#%%
import torch
from torch.optim import Adam
from transformers.configuration_albert import AlbertConfig
from transformers.configuration_bert import BertConfig
from src.dataloader.Dataset import EETaskDataloader
from src.dataloader.utils import load_data
from src.model.AlbertCRF import AlbertCrfForNer
from src.model.BertCRF import BertCrfForNer
from src.model.BertSoftMax import BertSoftmaxForNer
from src.util.EETaskRun import Run
from src.util.extract_arguments import extract_arguments_crf,extract_arguments_softmax
from src.util.utils import lcs
#%%
config = BertConfig.from_pretrained(
    r"/home/longred/lic2020_baselines/chinese_L-12_H-768_A-12/bert-base-chinese-config.json")
config.pretrained_path = r"/home/longred/lic2020_baselines/chinese_L-12_H-768_A-12/bert-base-chinese-pytorch_model.bin"
config.vocab_path = r"/home/longred/lic2020_baselines/chinese_L-12_H-768_A-12/vocab.txt"
config.train_data_path = r"/home/longred/EETask/data/train.json"
config.batch_size = 32
config.event_schema_path = r"/home/longred/EETask/data/event_schema.json"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EE = EETaskDataloader(config)
train_loader = EE.get_train_data_loader()
config.num_labels = EE.num_labels
config.label2id = EE.label2id
data = load_data("/home/longred/EETask/data/dev.json")
model = BertSoftmaxForNer.from_pretrained(
    pretrained_model_name_or_path=config.pretrained_path,
    config=config).to(device)
#%%
# To reproduce BertAdam specific behavior set correct_bias=False
from transformers import AdamW,get_linear_schedule_with_warmup
lr = 1e-3
max_grad_norm = 1.0
num_training_steps = 374
num_warmup_steps = 100
warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1
# optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
optimizer = AdamW([
    {'params': model.bert.parameters(),'lr':0.0001},
    {'params': model.classifier.parameters(),'lr':0.001},
    # {'params': model.crf.parameters(), 'lr': 0.001}
    ],lr=lr,correct_bias=False)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
#%%
r = Run()
r.device = device
r.dev_data =data
r.extract_arguments = extract_arguments_softmax
r.id2label = EE.id2label
r.label2id = EE.label2id
r.net = model
r.optim = optimizer
r.scheduler = scheduler
r.tokenizer = EE.tokenizer
r.train_loader = EE.get_train_data_loader()
#%%
# r.train()
# #%%
# r.evaluate()
#%%

#%%

best_f1 = 0
e_t = 0
while (True):
    r.train()
    t = r.evaluate()
    if t[0]>best_f1:
        best_f1 = t[0]
        e_t = 0
        torch.save(r.net.state_dict(),
                   "/home/longred/EETask/data/BertSoftMaxForNer_small.bin")
        print("save model {}".format(best_f1))
    e_t += 1
    if e_t>10:
        break


#%%



# %%
