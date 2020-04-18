#%%
import os
import sys
sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
#%%
import json
f = open("/home/longred/EETask/data/train.json")
f = list(f)
data = json.loads(f[0])
#%%
data
#%%
def load_data(filepath):
    D = []
    with open(filepath, encoding="UTF-8") as f:
        for l in f:
            l = json.loads(l)
            arguments = {}
            for event in l['event_list']:
                for argument in event['arguments']:
                    key = argument['argument']
                    value =  argument['role']
                    arguments[key] = value
                D.append((l['text'], l['event_type'], arguments))
    return D
#%%

# %%
temp = []
sum = 0
sum_all = 0
with open("/home/longred/EETask/data/train.json", encoding="UTF-8") as f:
    for l in f:
        l = json.loads(l)
        arguments = {}
        n = 0
        if len(l['event_list'])>1:
            print(l['event_list'][0]['event_type'],
                  l['event_list'][1]['event_type'])
            sum+=1
        sum_all +=1
#%%
sum / sum_all
# %%
event_schema = open("/home/longred/EETask/data/event_schema.json")
event = {}
for i in event_schema:
    data = json.loads(i)
    role = []
    for j in data["role_list"]:
        role.append(j['role'])
    event[data['event_type']] = role
#%%
event
# %%
len(set(role))

# %%
def get_data_info(event_schema_path):
    with open(event_schema_path, encoding="UTF-8") as f:
        id2label, label2id, n = {}, {}, 1
        id2label[0] = "[PAD]"
        label2id["[PAD]"] = 0
        for l in f:
            l = json.loads(l)
            for role in l['role_list']:
                key = role['role']
                if key in label2id:
                    pass
                else:
                    id2label[n] = key
                    label2id[key] = n
                    n += 1

        num_labels = len(id2label)

    return num_labels, id2label, label2id


#%%
num_labels, id2label, label2id = get_data_info(
    "/home/longred/EETask/data/event_schema.json")
#%%
num_labels
id2label
# %%
len(label2id)

# %%
data
