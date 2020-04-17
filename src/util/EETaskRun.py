import torch
from tqdm import tqdm
import json
from src.util.utils import lcs


class Run:
    def __init__(self):
        super().__init__()
        self.device = None
        self.optim = None
        self.writer = None
        self.net = None
        self.dev_data = None
        self.extract_arguments = None
        self.tokenizer = None
        self.id2label = None
        self.label2id = None
        self.train_loader = None
        self.scheduler = None

    def train(self):
        self.net.train()
        device = self.device
        loader = tqdm(self.train_loader)
        for i in loader:
            self.optim.zero_grad()
            loss, out = self.net(input_ids=i.input_ids.to(device),
                                 attention_mask=i.attention_mask.to(device),
                                 token_type_ids=i.token_type_ids.to(device),
                                 position_ids=None,
                                 head_mask=None,
                                 labels=i.label_ids.to(device),
                                 input_lens=i.seq_len)
            loss.backward()
            # print(loss)
            # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            self.optim.step()
            self.scheduler.step()
            loader.set_postfix(loss=loss.item())

    def evaluate(self):
        self.net.eval()
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for text, arguments in tqdm(self.dev_data):
            inv_arguments = {v: k for k, v in arguments.items()}
            pred_arguments = self.extract_arguments(
                self.net, text, self.tokenizer, self.id2label)
            pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
            Y += len(pred_inv_arguments)
            Z += len(inv_arguments)
            for k, v in pred_inv_arguments.items():
                if k in inv_arguments:
                    # 用最长公共子串作为匹配程度度量
                    l = lcs(v, inv_arguments[k])
                    X += 2. * l / (len(v) + len(inv_arguments[k]))
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        print("\n {0},  {1},  {2}".format(f1, precision, recall))
        return f1, precision, recall

    def predict_to_file(self, in_file, out_file):
        """预测结果到文件，方便提交
        """
        fw = open(out_file, 'w', encoding='utf-8')
        with open(in_file) as fr:
            for l in tqdm(fr):
                l = json.loads(l)
                arguments = self.extract_arguments(
                    self.net, l['text'], self.tokenizer, self.id2label)
                event_list = []
                for k, v in arguments.items():
                    event_list.append({
                        'event_type': v[0],
                        'arguments': [{
                            'role': v[1],
                            'argument': k
                        }]
                    })
                l['event_list'] = event_list
                # l.pop('text')
                l = json.dumps(l, ensure_ascii=False)
                fw.write(l + '\n')
        fw.close()
