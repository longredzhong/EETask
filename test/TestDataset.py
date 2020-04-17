#%%
import os
import sys
sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
#%%
from src.dataloader.utils import load_data
from src.util.tokenizers import Tokenizer
from src.util.utils import search
from src.dataloader.Dataset import get_data_info
#%%
data = load_data("/home/longred/EETask/data/train.json")
#%%
tokenizer = Tokenizer(
    "/home/longred/EETask/prev_trained_model/albert_tiny_zh/vocab.txt")
num_labels, id2label, label2id = get_data_info(
    "/home/longred/EETask/data/event_schema.json")
#%%
text,arguments = data[1]
#%%

input_ids, token_type_ids = tokenizer.encode(
    text, max_length=512)
seq_len = len(input_ids)
labels = [0] * seq_len
# labels[0] = label2id["[CLS]"]
# labels[-1] = label2id["[SEP]"]
attention_mask = [1]*seq_len
for argument in arguments.items():
    a_token_ids = tokenizer.encode(argument[0])[0][1:-1]
    start_index = search(a_token_ids, input_ids)
    if start_index != -1:
        labels[start_index] = label2id[argument[1]]
        for i in range(1, len(a_token_ids)):
            labels[start_index + i] = label2id[argument[1]]

#%%
text
#%%
labels

# %%
arguments

# %%
len(label2id)

# %%
token_type_ids


# %%
