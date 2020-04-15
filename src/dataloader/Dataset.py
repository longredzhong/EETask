from torchtext.data import BucketIterator, Dataset, Example, Field
from src.dataloader.utils import load_data
from src.util.utils import search
import json
from src.util.tokenizers import Tokenizer

max_length = 512


def get_data_info(event_schema_path):
    with open(event_schema_path, encoding="UTF-8") as f:
        id2label, label2id, n = {}, {}, 0
        for l in f:
            l = json.loads(l)
            for role in l['role_list']:
                key = (l['event_type'], role['role'])
                id2label[n] = key
                label2id[key] = n
                n += 1
        num_labels = len(id2label) * 2 + 1

        id2label[n] = "[CLS]"
        label2id["[CLS]"] = n

        id2label[n] = "[SEP]"
        label2id["[SEP]"] = n

    return num_labels+2, id2label, label2id


class EETaskDataset(Dataset):
    def __init__(self, path, fields, tokenizer, label2id):
        examples = []
        data = load_data(path)
        for (text, arguments) in data:
            input_ids, token_type_ids = tokenizer.encode(
                text, max_length=max_length)

            seq_len = len(input_ids)
            labels = [0] * seq_len
            labels[0] = label2id["[CLS]"] * 2 + 1
            labels[-1] = label2id["[SEP]"] * 2 + 2
            attention_mask = [1]*seq_len
            for argument in arguments.items():
                a_token_ids = tokenizer.encode(argument[0])[0][1:-1]
                start_index = search(a_token_ids, input_ids)
                # if start_index != -1:
                #     for i in range(0, len(a_token_ids)):
                #         labels[start_index + i] = label2id[argument[1]]
                if start_index != -1:
                    labels[start_index] = label2id[argument[1]] * 2+ 1
                    for i in range(1, len(a_token_ids)):
                        labels[start_index + i] = label2id[argument[1]] * 2+ 2


            assert len(input_ids) == len(token_type_ids) == len(
                attention_mask) == len(labels) == seq_len

            examples.append(Example.fromlist([
                input_ids, token_type_ids, attention_mask, labels, seq_len
            ], fields))
        super().__init__(examples, fields)


class EETaskDataloader:
    def __init__(self, config):
        self.config = config


        self.tokenizer = Tokenizer(self.config.vocab_path)

        self.num_labels, self.id2label, self.label2id = get_data_info(self.config.event_schema_path)
        from torchtext.data import RawField
        from src.dataloader.utils import sequence_padding
        self.fields = [
            ("input_ids", RawField(postprocessing=sequence_padding)),
            ("token_type_ids", RawField(postprocessing=sequence_padding)),
            ("attention_mask", RawField(postprocessing=sequence_padding)),
            ("label_ids", RawField(postprocessing=sequence_padding)),
            ("seq_len", RawField())
        ]

    def get_train_data_loader(self):
        self.traindataset = EETaskDataset(self.config.train_data_path, self.fields, self.tokenizer, self.label2id)
        self.trainloader = BucketIterator(self.traindataset, batch_size=self.config.batch_size,
                                          sort_key=lambda x: x.seq_len,sort=True,
                                          sort_within_batch=True, train=True, shuffle=True)
        return self.trainloader

#%%
if __name__ == '__main__':
    config = classmethod(-1)
    config.vocab_path = r"\\wsl$\Ubuntu\home\longred\EETask\prev_trained_model\albert_tiny_zh\vocab.txt"
    config.train_data_path = r"\\wsl$\Ubuntu\home\longred\EETask\data\train.json"
    config.batch_size = 32
    config.event_schema_path = r"\\wsl$\Ubuntu\home\longred\EETask\data\event_schema.json"
    EE = EETaskDataloader(config)
    #%%
    t = EE.get_train_data_loader()
    #%%
    t = iter(t)
    #%%
    next(t)


# %%
