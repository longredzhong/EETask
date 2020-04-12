# %%
from src.model.AlbertModel import AlbertTokenClassification
from src.util.Run import Run_model
from src.dataProcess.dataset import num_labels
from src.util.utils import load_config_from_json
import torch

config = load_config_from_json(
    r"\\wsl$\Ubuntu\home\longred\EETask\prev_trained_model\albert_tiny_zh\albert_config.json")
config.vocab_path = r"\\wsl$\Ubuntu\home\longred\EETask\prev_trained_model\albert_tiny_zh\vocab.txt"
config.train_data_path = r"\\wsl$\Ubuntu\home\longred\EETask\data\train.json"
config.dev_data_path = r"\\wsl$\Ubuntu\home\longred\EETask\data\dev.json"
config.num_labels = num_labels

config.model_name = r"model.bin"
config.batch_size = 32
net = AlbertTokenClassification(config=config)

# %%
state_dict = torch.load(
    r"\\wsl$\Ubuntu\home\longred\EETask\prev_trained_model\albert_tiny_zh\pytorch_model.bin")
net.load_state_dict(state_dict, strict=False)
# %%
# %%
t = Run_model(net, config)
# %%
t.net.to(t.device)
t.train(t.train_dataloader)

# %%
t.net.cpu()
t.net.eval()
t.evaluate(t.dev_data)

# %%
t.config.batch_size = 32
t.get_dataset()
# %%
len(t.train_dataloader)

# %%
for i in range(50):
    t.net.to(t.device)
    t.train(t.train_dataloader)
    t.net.cpu()
    t.net.eval()
    f1, pre, recall = t.evaluate(t.dev_data)
    print(f1, pre, recall)

# %%
t.net.cpu()
t.extract_arguments("7岁男孩地震遇难，父亲等来噩耗失声痛哭，网友：希望不再有伤亡")

# %%
token_ids, _ = t.tokenizer.encode(r"希望不再有伤亡")
token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1)
nodes = t.net(token_ids)
# %%
i, j, k = next(iter(t.train_dataloader))

# %%
k[1]

# %%
i

# %%
