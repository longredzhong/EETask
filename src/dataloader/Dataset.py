from torchtext.data import BucketIterator, Dataset, Example, Field
from src.dataloader.utils import load_data
from src.util.utils import search
import json

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
        num_labels = len(id2label) + 1
    return num_labels, id2label, label2id


class EETaskDataset(Dataset):
    def __init__(self, path, fields, tokenizer, label2id):
        examples = []
        data = load_data(path)
        token_ids_list, labels_list = [], []
        for (text, arguments) in data:
            token_ids, segment_ids = tokenizer.encode(text, max_length=max_length)
            labels = [0] * len(token_ids)
            for argument in arguments.items():
                a_token_ids = tokenizer.encode(argument[0])[0][1:-1]
                start_index = search(a_token_ids, token_ids)
                if start_index != -1:
                    labels[start_index] = label2id[argument[1]] + 1
                    for i in range(1, len(a_token_ids)):
                        labels[start_index + i] = label2id[argument[1]] + 1
            examples.append(Example.fromlist([
                            token_ids, labels,len(token_ids)
                            ], fields))
        super().__init__(examples, fields)


class EETaskDataloader:
    def __init__(self,data_path):
        from src.util.tokenizers import Tokenizer

        tokenizer = Tokenizer("prev_trained_model/albert_tiny_zh/vocab.txt")

        num_labels, id2label, label2id = get_data_info("data/event_schema.json")
        from torchtext.data import RawField
        from src.dataloader.utils import sequence_padding
        fields = [
            ("input", RawField(postprocessing=sequence_padding)),
            ("target", RawField(postprocessing=sequence_padding)),
            ("len", RawField())
        ]
        dataset = EETaskDataset("data/train.json", fields, tokenizer, label2id)
        loader = BucketIterator(dataset, batch_size=32, sort_key=lambda x: x.len, sort=True, sort_within_batch=True,
                            train=True, shuffle=True)

# %%








