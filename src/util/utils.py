import json
import copy


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