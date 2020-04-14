import torch
import torch.nn as nn
# from Albert import AlbertModel
from pytorchcrf import CRF
from transformers import AlbertModel, AlbertTokenizer,AlbertPreTrainedModel
from src.util.utils import seq_len_to_mask
from torch.nn import CrossEntropyLoss

class AlbertCRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.albert = AlbertModel.from_pretrained(config.pretrained_path)
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
        return output

    def forward(self, input_ids, seq_len, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        output = self._forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               position_ids=position_ids, head_mask=head_mask)
        mask = seq_len_to_mask(seq_len, batch_first=True)
        output = self.crf.decode(output, mask=mask)
        return output

    def loss(self, x, seq_len, y, attention_mask=None, token_type_ids=None,
             position_ids=None, head_mask=None, reduction='mean'):
        mask = seq_len_to_mask(seq_len, batch_first=True)
        emissions = self._forward(x, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                  position_ids=position_ids, head_mask=head_mask)
        return self.crf(emissions, y, mask=mask,reduction=reduction)


class AlbertSoftmaxForNer(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        loss_fct = CrossEntropyLoss(ignore_index=0)
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)
if __name__ == '__main__':
    config = classmethod(-1)
    config.num_labels = 5
    config.pretrained_path = r"\\wsl$\Ubuntu\home\longred\EETask\prev_trained_model\albert_tiny_zh"
    model = AlbertCRF(config=config)

    # %%
