import torch
import torch.nn as nn
# from Albert import AlbertModel
from transformers import BertModel,BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from src.model.crf import CRF


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(tagset_size=config.num_labels,
                       tag_dictionary=config.label2id, is_bert=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, input_lens=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf.calculate_loss(
                logits, tag_list=labels, lengths=input_lens)
            outputs = (loss,)+outputs
        return outputs  # (loss), scores
