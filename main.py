#%%
from src.model.AlbertCRF import AlbertCRF
from src.dataloader.Dataset import EETaskDataloader
from torch.optim import Adam
import torch
#%%
if __name__ == '__main__':
    #%%
    config = classmethod(-1)
    config.vocab_path = r"\\wsl$\Ubuntu\home\longred\EETask\prev_trained_model\albert_tiny_zh\vocab.txt"
    config.train_data_path = r"\\wsl$\Ubuntu\home\longred\EETask\data\train.json"
    config.batch_size = 32
    config.event_schema_path = r"\\wsl$\Ubuntu\home\longred\EETask\data\event_schema.json"
    config.pretrained_path = r"\\wsl$\Ubuntu\home\longred\EETask\prev_trained_model\albert_tiny_zh"
    EE = EETaskDataloader(config)
    train_loader = EE.get_train_data_loader()
    #%%
    config.num_labels = EE.num_labels
    config.hidden_size = 312
    net = AlbertCRF(config)
    #%%
    from torch.optim import SGD
    optim = Adam(net.parameters(),lr=0.001)
    #%%
    ttt = iter(train_loader)
    #%%
    batch = next(ttt)
    #%%
    optim.zero_grad()
    loss = net.loss(batch.input,torch.tensor(batch.len),batch.target)
    loss.backward()
    optim.step()
    print(loss.item())
    #%%
    batch.target.size()
    #%%
    net.forward(batch.input,torch.tensor(batch.len))
    #%%
    torch.max(net._forward(batch.input),2)[1]
    #%%
    net._forward(batch.input)
    #%%
    batch.target.size()
    #%%
    for i in train_loader:
        optim.zero_grad()
        loss = net.loss(i.input, torch.tensor(i.len), i.target)
        loss.backward()
        optim.step()
        print(loss.item())

