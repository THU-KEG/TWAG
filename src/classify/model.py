import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertTokenizer, AlbertModel


class AlbertClassifierModel(nn.Module):
    def __init__(self, num_topics=5, dropout=0.1,
                 albert_model_dir="/data1/tsq/TWAG/data/pretrained_models/albert"):
        super(AlbertClassifierModel, self).__init__()
        self.ntopic = num_topics
        # self.albert_model = AlbertModel.from_pretrained('albert-base-v2')
        try:
            self.albert_model = AlbertModel.from_pretrained(albert_model_dir)
        except OSError:
            model = AlbertModel.from_pretrained('albert-base-v2')
            model.save_pretrained(albert_model_dir)
            self.albert_model = model

        self.fc_layer = nn.Linear(768, self.ntopic)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, use_cls=False):
        if (use_cls):
            cls_hidden = self.albert_model(input_ids=input_ids, attention_mask=attention_mask)[1]
            y = self.fc_layer(cls_hidden)
        else:
            all_hidden = self.albert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            mean_hidden = torch.mean(all_hidden, dim=1, keepdim=False)
            y = self.fc_layer(mean_hidden)
        return y

    def predict(self, input_ids, attention_mask, use_cls=False):
        if (use_cls):
            cls_hidden = self.albert_model(input_ids=input_ids, attention_mask=attention_mask)[1]
            y = self.fc_layer(cls_hidden)
        else:
            all_hidden = self.albert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            mean_hidden = torch.mean(all_hidden, dim=1, keepdim=False)
            y = self.fc_layer(mean_hidden)

        predict_y = torch.softmax(y, dim=1)
        return predict_y


if __name__ == "__main__":
    pass
