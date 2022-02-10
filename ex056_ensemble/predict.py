#!/usr/bin/env python
# coding: utf-8

# # Overview
# This is kernel is almost the same as [Lightweight Roberta solution in PyTorch](https://www.kaggle.com/andretugan/lightweight-roberta-solution-in-pytorch), but instead of "roberta-base", it starts from [Maunish's pre-trained model](https://www.kaggle.com/maunish/clrp-roberta-base).
#
# Acknowledgments: some ideas were taken from kernels by [Torch](https://www.kaggle.com/rhtsingh) and [Maunish](https://www.kaggle.com/maunish).


import os
import math
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup

from sklearn.model_selection import KFold

import gc
gc.enable()

# base_path = os.path.dirname(os.path.abspath(__file__))
base_path = '.'
# m_path = f'{base_path}/model'
m_path1 = '../input/commonlit-ex005-robertamodel'
m_path2 = '../input/commonlit-ex026-roberta-large'
m_path3 = '../input/commonlit-ex055-roberta-large'
m_path4 = '../input/commonlit-ex047-roberta-large'


NUM_FOLDS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 8
MAX_LEN_256 = 256
MAX_LEN_300 = 300
EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]
# ROBERTA_PATH = base_path + \
#     "/../ex010_roberta_large_pretrain/clrprobertalarge/clrp_roberta_large"
# TOKENIZER_PATH = base_path + \
#     "/../ex010_roberta_large_pretrain/clrprobertalarge/clrp_roberta_large"
ROBERTA_PATH_BASE = base_path + "/../input/clrp-roberta-base/clrp_roberta_base"
TOKENIZER_PATH_BASE = base_path + "/../input/clrp-roberta-base/clrp_roberta_base"
ROBERTA_PATH_LARGE = base_path + "/../input/clrprobertalarge/clrp_roberta_large"
TOKENIZER_PATH_LARGE = base_path + "/../input/clrprobertalarge/clrp_roberta_large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True


train_df = pd.read_csv(
    base_path + "/../input/commonlitreadabilityprize/train.csv")

# Remove incomplete entries if any.
train_df.drop(train_df[(train_df.target == 0) & (train_df.standard_error == 0)].index,
              inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(
    base_path + "/../input/commonlitreadabilityprize/test.csv")
submission_df = pd.read_csv(
    base_path + "/../input/commonlitreadabilityprize/sample_submission.csv")


tokenizer_base = AutoTokenizer.from_pretrained(TOKENIZER_PATH_BASE)
tokenizer_large = AutoTokenizer.from_pretrained(TOKENIZER_PATH_LARGE)


# # Dataset
class LitDataset256(Dataset):
    def __init__(self, df, tokenizer, inference_only=False):
        super().__init__()

        self.df = df
        self.inference_only = inference_only
        self.text = df.excerpt.tolist()
        #self.text = [text.replace("\n", " ") for text in self.text]

        if not self.inference_only:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)

        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding='max_length',
            max_length=MAX_LEN_256,
            truncation=True,
            return_attention_mask=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])

        if self.inference_only:
            return (input_ids, attention_mask)
        else:
            target = self.target[index]
            return (input_ids, attention_mask, target)


# # Model
# The model is inspired by the one from [Maunish](https://www.kaggle.com/maunish/clrp-roberta-svm).
class LitModel1(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(ROBERTA_PATH_BASE)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.roberta = AutoModel.from_pretrained(
            ROBERTA_PATH_BASE, config=config)

        self.attention = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(768, 1)
        )

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)

        # There are a total of 13 layers of hidden states.
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        # The number of cells is MAX_LEN.
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # In order to condense hidden states of all cells to a context vector,
        # we compute a weighted average of the hidden states of all cells.
        # We compute the weight of each cell, using the attention neural network.
        weights = self.attention(last_layer_hidden_states)

        # weights.shape is BATCH_SIZE x MAX_LEN x 1
        # last_layer_hidden_states.shape is BATCH_SIZE x MAX_LEN x 768
        # Now we compute context_vector as the weighted average.
        # context_vector.shape is BATCH_SIZE x 768
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)

        # Now we reduce the context vector to the prediction score.
        return self.regressor(context_vector)


class LitModel2(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(ROBERTA_PATH_LARGE)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.roberta = AutoModel.from_pretrained(
            ROBERTA_PATH_LARGE, config=config)

        self.attention = nn.Sequential(
            # nn.Linear(768, 512),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.lstm = nn.Sequential(
            nn.LSTM(1024, 768, batch_first=True, bidirectional=False)
        )

        self.regressor = nn.Sequential(
            nn.Linear(768, 1)
        )

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)

        # There are a total of 13 layers of hidden states.
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.

        # 最終層 (4, 256, 1024)
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        # The number of cells is MAX_LEN.
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # In order to condense hidden states of all cells to a context vector,
        # we compute a weighted average of the hidden states of all cells.
        # We compute the weight of each cell, using the attention neural network.
        weights = self.attention(last_layer_hidden_states)  # [4, 256, 1]

        # weights.shape is BATCH_SIZE x MAX_LEN x 1
        # last_layer_hidden_states.shape is BATCH_SIZE x MAX_LEN x 768
        # Now we compute context_vector as the weighted average.
        # context_vector.shape is BATCH_SIZE x 768
        context_vector = weights * last_layer_hidden_states  # (4, 256, 1024)

        # LSTM
        lstm_output, (h, c) = self.lstm(context_vector)  # (4, 256, 768)

        context_vector = torch.sum(lstm_output, dim=1)  # [4, 768]

        # Now we reduce the context vector to the prediction score.
        return self.regressor(context_vector)  # [4, 1]


class LitModel3(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(ROBERTA_PATH_LARGE)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.roberta = AutoModel.from_pretrained(
            ROBERTA_PATH_LARGE, config=config)

        self.layer_norm = nn.LayerNorm(1024)

        self.regressor = nn.Sequential(
            nn.Linear(1024, 1)
        )

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)

        # There are a total of 13 layers of hidden states.
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.

        # 最終層 (8, 256, 1024)
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(last_layer_hidden_states.size()).float()  # (8, 256, 1024)
        sum_embeddings = torch.sum(
            last_layer_hidden_states * input_mask_expanded, 1)  # (8, 1024)
        sum_mask = input_mask_expanded.sum(1)  # (8, 1024)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # (8, 1024)
        mean_embeddings = sum_embeddings / sum_mask  # (8, 1024)

        norm_mean_embeddings = self.layer_norm(mean_embeddings)

        # Now we reduce the context vector to the prediction score.
        return self.regressor(norm_mean_embeddings)  # [8, 1]


class LitModel4(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(ROBERTA_PATH_LARGE)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.roberta = AutoModel.from_pretrained(
            ROBERTA_PATH_LARGE, config=config)

        self.attention = nn.Sequential(
            # nn.Linear(768, 512),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.lstm1 = nn.Sequential(
            nn.LSTM(1024, 768, batch_first=True, bidirectional=False)
        )

        self.lstm2 = nn.Sequential(
            nn.LSTM(768, 512, batch_first=True, bidirectional=False)
        )

        self.regressor = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)

        # There are a total of 13 layers of hidden states.
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.

        # 最終層 (4, 256, 1024)
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        # The number of cells is MAX_LEN.
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # In order to condense hidden states of all cells to a context vector,
        # we compute a weighted average of the hidden states of all cells.
        # We compute the weight of each cell, using the attention neural network.
        weights = self.attention(last_layer_hidden_states)  # [4, 256, 1]

        # weights.shape is BATCH_SIZE x MAX_LEN x 1
        # last_layer_hidden_states.shape is BATCH_SIZE x MAX_LEN x 768
        # Now we compute context_vector as the weighted average.
        # context_vector.shape is BATCH_SIZE x 768
        context_vector = weights * last_layer_hidden_states  # (4, 256, 1024)

        # LSTM
        lstm_output, (h, c) = self.lstm1(context_vector)  # (4, 256, 768)
        lstm_output, (h, c) = self.lstm2(lstm_output)  # (4, 256, 512)

        context_vector = torch.sum(lstm_output, dim=1)  # [4, 512]

        # Now we reduce the context vector to the prediction score.
        return self.regressor(context_vector)  # [4, 1]


def predict(model, data_loader):
    """Returns an np.array with predictions of the |model| on |data_loader|"""
    model.eval()

    result = np.zeros(len(data_loader.dataset))
    index = 0

    with torch.no_grad():
        for batch_num, (input_ids, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            pred = model(input_ids, attention_mask)

            result[index: index + pred.shape[0]] = pred.flatten().to("cpu")
            index += pred.shape[0]

    return result


# # Inference
# predict1
test_dataset = LitDataset256(test_df, tokenizer_base, inference_only=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         drop_last=False, shuffle=False, num_workers=2)
all_predictions1 = np.zeros((NUM_FOLDS, len(test_df)))

for index in range(NUM_FOLDS):
    model_path = f"{m_path1}/model_{index + 1}.pth"
    print(f"\nUsing {model_path}")

    model = LitModel1()
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)

    all_predictions1[index] = predict(model, test_loader)

    del model
    gc.collect()


# predict2
test_dataset = LitDataset256(test_df, tokenizer_large, inference_only=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         drop_last=False, shuffle=False, num_workers=2)
all_predictions2 = np.zeros((NUM_FOLDS, len(test_df)))

for index in range(NUM_FOLDS):
    model_path = f"{m_path2}/model_{index + 1}.pth"
    print(f"\nUsing {model_path}")

    model = LitModel2()
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)

    all_predictions2[index] = predict(model, test_loader)

    del model
    gc.collect()


# predict3
test_dataset = LitDataset256(test_df, tokenizer_large, inference_only=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         drop_last=False, shuffle=False, num_workers=2)
all_predictions3 = np.zeros((NUM_FOLDS, len(test_df)))

for index in range(NUM_FOLDS):
    model_path = f"{m_path3}/model_{index + 1}.pth"
    print(f"\nUsing {model_path}")

    model = LitModel3()
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)

    all_predictions3[index] = predict(model, test_loader)

    del model
    gc.collect()

# predict4
test_dataset = LitDataset256(test_df, tokenizer_large, inference_only=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         drop_last=False, shuffle=False, num_workers=2)
all_predictions4 = np.zeros((NUM_FOLDS, len(test_df)))

for index in range(NUM_FOLDS):
    model_path = f"{m_path4}/model_{index + 1}.pth"
    print(f"\nUsing {model_path}")

    model = LitModel4()
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)

    all_predictions4[index] = predict(model, test_loader)

    del model
    gc.collect()


predictions1 = all_predictions1.mean(axis=0)
predictions2 = all_predictions2.mean(axis=0)
predictions3 = all_predictions3.mean(axis=0)
predictions4 = all_predictions4.mean(axis=0)

final_prediction = predictions1 * 0.25 + predictions2 * 0.25 + \
    predictions3 * 0.25 + predictions4 * 0.25

submission_df.target = final_prediction
print(submission_df)
submission_df.to_csv(base_path + "/submission.csv", index=False)
