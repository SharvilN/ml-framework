import transformers
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scipy

from sklearn import model_selection

class BertBaseUncased(nn.Module):
    
    def __init__(self, bert_path):
        super(BertBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 30)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(outputs[1])
        return self.out(bo)

class BertDatasetTraining:

    def __init__(self, qtitle, qbody, answer, tokenizer, targets, max_len):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = targets

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, index):
        question_title = str(self.qtitle[index])
        question_body = str(self.qbody[index])
        answer = str(self.answer[index])

        inputs = self.tokenizer.encode_plus(
            question_title + " " + question_body,
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        padding_len = self.max_len - len(ids)
        ids = ids + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index, :], dtype=torch.float)
        }

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    step = 0
    for bi, d in enumerate(data_loader):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        if step%10 == 0: print(f'BCEloss={loss}, step={step}')
        loss.backward()
        optimizer.step()
        step += 1
        if scheduler is not None:
            scheduler.step()

def eval_loop_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    for bi, d in enumerate(data_loader):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)

        fin_outputs.append(outputs.cpu().detach().numpy())
        fin_targets.append(targets.cpu().detach().numpy())

    return np.vstack(fin_outputs), np.vstack(fin_targets)

def run():
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    EPOCHS = 10

    dfx = pd.read_csv("inputs/google-quest-challenge/train.csv").fillna("none")
    df_train, df_valid = model_selection.train_test_split(dfx, test_size=0.1, random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    sample = pd.read_csv('inputs/google-quest-challenge/sample_submission.csv')
    target_cols = list(sample.drop('qa_id', axis=1).columns)
    train_targets = df_train[target_cols]
    valid_targets = df_valid[target_cols]

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = BertDatasetTraining(df_train.question_title.values,
    qbody=df_train.question_body.values,
    answer=df_train.answer.values,
    tokenizer=tokenizer,
    targets=train_targets.values,
    max_len=512)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True)

    valid_dataset = BertDatasetTraining(df_valid.question_title.values,
    qbody=df_valid.question_body.values,
    answer=df_valid.answer.values,
    tokenizer=tokenizer,
    targets=valid_targets,
    max_len=MAX_LEN)

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
    batch_size=4,
    shuffle=False)

    device = 'cpu'
    lr = 3e-5
    num_train_steps = int(len(train_dataset)/TRAIN_BATCH_SIZE * EPOCHS)
    print(f"Total steps: {num_train_steps}")
    model = BertBaseUncased('bert-base-uncased').to(device)
    optimizer = transformers.AdamW(model.parameters(), lr=lr)

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    for epoch in range(EPOCHS):
        train_loop_fn(train_dataloader, model, optimizer, device, scheduler=scheduler)
        o, t = eval_loop_fn(valid_dataloader, model, device)

        spear = []
        for jj in range(t.shape[1]):
            p1 = list(t[:, jj])
            p2 = list(o[:, jj])
            coef, _ = np.nan_to_num(scipy.stats.spearmanr(p1, p2))
            spear.append(coef)

        spear = np.mean(spear)    
        print(f"Epoch : {epoch} , spearman = {spear}")
        torch.save(mode.state_dict(), 'model.bin')

if __name__ == '__main__':
    # print(torch.cuda.is_available())
    run()

