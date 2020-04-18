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
max_len = 0
len_text = []
for text,au in data:
    len_text.append(len(text))
#%%
max_len


# %%
max(len_text)

# %%
sum(len_text)/len(len_text)

# %%
import matplotlib.pyplot as plt

# %%
plt.hist(len_text)


# %%
