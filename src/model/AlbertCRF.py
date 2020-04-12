import torch
import torch.nn as nn
# from Albert import AlbertModel
from pytorchcrf import CRF
from transformers import AlbertModel, AlbertTokenizer


class AlbertCRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.albert = AlbertModel.from_pretrained(
            r"\\wsl$\Ubuntu\home\longred\EETask\prev_trained_model\albert_tiny_zh")
        self.classifier = nn.Linear(312, config.num_labels)
        self.crf = CRF(config.num_labels)
        self.pad_index = config.pad_index

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        output = self.albert(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)
        sequence_output = output[0]
        output = self.classifier(sequence_output)
        mask = torch.ne(input_ids, self.pad_index)
        output = self.crf.decode(output, mask=mask)
        return output

    def loss(self, x, sent_lengths, y):
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, sent_lengths)
        return self.crflayer(emissions, y, mask=mask)


if __name__ == '__main__':
    from Albert import AlbertConfig

    config = AlbertConfig()
    config.num_labels = 5
    config.pad_index = -1
    model = AlbertCRF(config=config)

