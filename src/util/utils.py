import json
import copy
import numpy as np
import torch


def load_config_from_json(json_file_path):
    config = classmethod(-1)
    with open(json_file_path, "r", encoding='utf-8') as reader:
        json_object = json.load(reader)
    for key, value in json_object.items():
        config.__dict__[key] = value
    return config


def save_config_to_json(config, json_file_path):
    output = copy.deepcopy(config.__dict__)
    json_string = json.dumps(output, indent=2, sort_keys=True) + "\n"
    with open(json_file_path, "w", encoding='utf-8') as writer:
        writer.write(json_string)

def lcs(str_a, str_b):
    """
    longest common subsequence of str_a and str_b, with O(n) space complexity
    """
    if len(str_a) == 0 or len(str_b) == 0:
        return 0
    dp = [0 for _ in range(len(str_b) + 1)]
    for i in range(1, len(str_a) + 1):
        left_up = 0
        dp[0] = 0
        for j in range(1, len(str_b) + 1):
            left = dp[j-1]
            up = dp[j]
            if str_a[i-1] == str_b[j-1]:
                dp[j] = left_up + 1
            else:
                dp[j] = max([left, up])
            left_up = up
    return dp[len(str_b)]

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def seq_len_to_mask(seq_len, max_len=None,batch_first = True):
    r"""

    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    .. code-block::

        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)
        if not batch_first:
            mask = mask.swapaxes(0,1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1)).byte()
        if not batch_first:
            mask = mask.permute(1,0)
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

