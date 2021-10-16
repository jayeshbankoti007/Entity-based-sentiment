import torch.nn as nn
import config
import torch
from transformers import AutoModel


class BERTSentiment(nn.Module):
    def __init__(self):
        super(BERTSentiment, self).__init__()
        self.bert_layer = AutoModel.from_pretrained(config.MODEL_NAME)
        self.bert_drop_1 = nn.Dropout(0.4)
        self.dense_layer_1 = nn.Linear(768, 128)
        self.bert_drop_2 = nn.Dropout(0.4)
        self.dense_layer_2 = nn.Linear(128, 1)

    def forward(self, ids, mask, text_token_type, aspect_word_masking):

        bert_out_text = self.bert_layer(
            input_ids = ids,
            attention_mask = mask,
            token_type_ids = text_token_type)

        drop1_text = self.bert_drop_1(bert_out_text['last_hidden_state'])
        dense_text1 = self.dense_layer_1(drop1_text)

        drop2_text = self.bert_drop_2(dense_text1)
        dense_text2 = self.dense_layer_2(drop2_text)

        total_logits = torch.squeeze(dense_text2, -1)

        masked_array_lengths = torch.sum(aspect_word_masking, axis=1).tolist()

        selected_entity = torch.masked_select(total_logits, aspect_word_masking)
        selected_entity = torch.split(selected_entity, masked_array_lengths)

        cat_list = []
        for elements in selected_entity:
            cat_list.append(torch.mean(elements))

        sentiment_logits = torch.stack(cat_list, 0)

        return sentiment_logits
