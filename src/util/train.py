from tqdm import tqdm

def train(net, train_loader, optim,device,writer=None):
    loader = tqdm(train_loader)
    for i in loader:
        net.train()
        loss, out = net(input_ids=i.input_ids.to(device),
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
        if writer!=None:
            pass
            # writer.add(loss,i)
