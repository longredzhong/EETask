import torch
import torch.nn as nn
# from Albert import AlbertModel
from pytorchcrf import CRF
from transformers import AlbertModel, AlbertTokenizer
from src.util.utils import seq_len_to_mask


class AlbertCRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.albert = AlbertModel.from_pretrained(config.pretrained_path)
        self.softmax = nn.LogSoftmax(dim=2)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def _forward(self, input_ids, attention_mask=None, token_type_ids=None,
                 position_ids=None, head_mask=None):
        output = self.albert(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)
        sequence_output = output[0]
        output = self.softmax(self.classifier(sequence_output))
        return output

    def forward(self, input_ids, seq_len, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        output = self._forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               position_ids=position_ids, head_mask=head_mask)
        mask = seq_len_to_mask(seq_len, batch_first=True)
        output = self.crf.decode(output, mask=mask)
        return output

    def loss(self, x, seq_len, y,reduction='mean'):
        mask = seq_len_to_mask(seq_len, batch_first=True)
        emissions = self._forward(x)
        return self.crf(emissions, y, mask=mask,reduction=reduction)


if __name__ == '__main__':
    config = classmethod(-1)
    config.num_labels = 5
    config.pretrained_path = r"\\wsl$\Ubuntu\home\longred\EETask\prev_trained_model\albert_tiny_zh"
    model = AlbertCRF(config=config)

    # %%
