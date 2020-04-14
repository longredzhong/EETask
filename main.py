# %%
import os
import sys
sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))))
# %%
from src.model.AlbertCRF import AlbertCRF,AlbertSoftmaxForNer
from src.dataloader.Dataset import EETaskDataloader
from torch.optim import Adam
import torch
from src.util.utils import load_config_from_json
# %%
if __name__ == '__main__':
    # %%
    config = load_config_from_json(
        "/home/longred/EETask/prev_trained_model/albert_tiny_zh/albert_config.json")
    config.vocab_path = r"/home/longred/EETask/prev_trained_model/albert_tiny_zh/vocab.txt"
    config.train_data_path = r"/home/longred/EETask/data/train.json"
    config.batch_size = 32
    config.event_schema_path = r"/home/longred/EETask/data/event_schema.json"
    config.pretrained_path = r"/home/longred/EETask/prev_trained_model/albert_tiny_zh"
    EE = EETaskDataloader(config)
    train_loader = EE.get_train_data_loader()
    # %%
    config.num_labels = EE.num_labels
    config.hidden_size = 312
    net = AlbertSoftmaxForNer.from_pretrained(
        "/home/longred/EETask/prev_trained_model/albert_tiny_zh/", num_labels=config.num_labels)

    # %%
    optim = Adam(net.parameters(), lr=0.001)
    # %%
    for i in train_loader:
        net.zero_grad()
        loss,out = net(i.input_ids, attention_mask=i.attention_mask, token_type_ids=i.token_type_ids,
                   position_ids=None, head_mask=None, labels=i.label_ids)
        loss.backward()
        optim.step()
        print(loss.item())
    # %%
    a = torch.max(out,1)[1]
    # %%
    a.size()

    # %%
    out.size()

    # %%
    a[7]

# %%
