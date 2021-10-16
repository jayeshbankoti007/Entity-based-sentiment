from transformers import AutoTokenizer
import config
import torch
import numpy as np

class ExtractDataset:
    def __init__(self, sentences, sentiments, entity):
        self.sentences = list(map(self.remove_extra_space, sentences))
        self.sentiment_values = list(map(self.sentiment_encoder, sentiments))
        self.entity = list(map(self.remove_extra_space, entity))
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.sentence_encodings = self.tokenizer(self.sentences, return_offsets_mapping=True)
        self.max_len = config.MAX_LEN

    def sentiment_encoder(self, sentiment):
        if sentiment == 'positive':
            return 1
        else:
            return 0

    def remove_extra_space(self, text):
        return " ".join(text.split())

    def __len__(self):
        return len(self.entity)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        entity = self.entity[idx]
        sentiment_value = self.sentiment_values[idx]

        tok_sentence_ids = self.sentence_encodings.input_ids[idx]
        tok_sentence_offsets = self.sentence_encodings.offset_mapping[idx][1:-1]
        tok_sentence_type_id = self.sentence_encodings.token_type_ids[idx]
        tok_sentence_mask = self.sentence_encodings.attention_mask[idx]

        start_ids = [i for i in range(len(sentence)) if sentence.startswith(entity, i)]

        aspect_word_masking = np.zeros(len(tok_sentence_ids))

        word_counter = 0
        word_started = 0
        for i, (start_id, end_id) in enumerate(tok_sentence_offsets):
            if word_started:
                aspect_word_masking[i] = 1
                if start_ids[word_counter] + len(entity) == end_id:
                    word_counter += 1
                    word_started = 0
            else:
                if word_counter < len(start_ids) and start_ids[word_counter] == start_id:
                    word_started = 1
                    aspect_word_masking[i] = 1
                    if start_ids[word_counter] + len(entity) == end_id:
                        word_counter += 1
                        word_started = 0

        # Need to pad them 
        padding_len = self.max_len - len(tok_sentence_ids)

        tok_sentence_ids = tok_sentence_ids + [0] * padding_len
        tok_sentence_mask = tok_sentence_mask + [0] * padding_len
        tok_sentence_type_id = tok_sentence_type_id + [0] * padding_len
        aspect_word_masking = [0] + aspect_word_masking.tolist() + [0] + [0] * (padding_len-2)

        return {
            'input_ids' : torch.tensor(tok_sentence_ids, dtype=torch.long),
            'attention_mask' : torch.tensor(tok_sentence_mask, dtype=torch.long),
            'sentiment_value' : torch.tensor(sentiment_value, dtype=torch.float),
            'aspect_word_masking' : torch.tensor(aspect_word_masking, dtype=torch.bool),
            'token_type_ids' : torch.tensor(tok_sentence_type_id, dtype=torch.long)
        }


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(config.PROCESSED_TRAIN_FILE_PATH).dropna().reset_index(drop=True)

    dset = ExtractDataset(
        sentences = list(df.Sentence.values), 
        sentiments = list(df.Sentiment.values), 
        entity = list(df.Entity.values)
    )

    print(dset[0])