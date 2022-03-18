from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig, AdamW
from tqdm import tqdm
import torch
import pandas as pd
from collections import defaultdict
import sys

torch.manual_seed(42)

class TestTargetLearning:

    def __init__(self, model_name, steps=50, training=True):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_config(BertConfig())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.steps = steps
        self.model.train() if training else self.model.eval()
        self.loss = torch.nn.CrossEntropyLoss() if training else None
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5) if training else None

    def learn_target_words(self, data):

        target_words = data[0]["target_words"]
        context = data[0]["context"]

        input_sequence = self.tokenizer(context)
        attention_mask = torch.tensor(input_sequence["attention_mask"]).unsqueeze(0).to(self.device)
        masked_indices_targets = [context.split().index(target_word) + 1 for target_word in target_words]
        masked_indices_targets.sort()

        saved_values = defaultdict(list)

        for i in tqdm(range(self.steps)):

            target_loss = None
            target_trained_on_yet = None

            for i, word in enumerate(context.split()):

                masked_index = i + 1
                input_ids = torch.tensor(input_sequence["input_ids"]).unsqueeze(0).to(self.device)

                if masked_index in masked_indices_targets:
                    if target_trained_on_yet is None:
                        # prepare masking for multiple masks
                        targets_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(target_words)).to(self.device)
                        input_ids[0, masked_indices_targets] = self.tokenizer.mask_token_id
                        outputs = self.model(input_ids, attention_mask=attention_mask)

                        logits = outputs[0][:, masked_indices_targets].squeeze(0)
                        loss = self.loss(logits, targets_ids)
                        target_trained_on_yet = loss
                        saved_values[f"{word}_{masked_index}"].append(float(loss))
                        loss.backward()
                    else:
                        saved_values[f"{word}_{masked_index}"].append(float(target_trained_on_yet))
                else:
                    input_ids[0, masked_index] = self.tokenizer.mask_token_id
                    target_id = torch.tensor(self.tokenizer.convert_tokens_to_ids([word])).to(self.device)[0]
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs[0][:, masked_index]
                    loss = self.loss(logits, target_id.unsqueeze(0))
                    saved_values[f"{word}_{masked_index}"].append(float(loss))

                self.optimizer.step()
                self.optimizer.zero_grad()

        loss_for_words = pd.DataFrame.from_dict(saved_values)

        return loss_for_words

    def learn_target_word_batch(self, data):  # TODO
        contexts = [sample["context"] for sample in data]
        targets = [sample["target_word"] for sample in data]
        masked_index_targets = [context.split().index(target_word) + 1 for context in contexts]

        input_sequences = self.tokenizer(contexts, padding=True)
        attention_masks = torch.tensor(input_sequences["attention_mask"]).to(self.device)
        saved_values = defaultdict(list)

        for i in tqdm(range(self.steps)):

            target_loss = None
            input_ids = torch.tensor(input_sequences["input_ids"]).to(self.device)
            input_ids[list(range(len(masked_index_targets))), masked_index_targets] = self.tokenizer.mask_token_id
            target_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(targets)).to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_masks)
            logits = outputs["logits"][:,:,masked_index_targets]
            loss = self.loss(outputs["logits"], target_ids)
            saved_values[f"{word}_{masked_index}"].append(float(loss))
            if masked_index == masked_index_target:
                target_loss = loss

            target_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        loss_for_words = pd.DataFrame.from_dict(saved_values)

        return loss_for_words

    def learn_target_word_single(self, data):
        target_word = data[0]["target_word"]
        context = data[0]["context"]
        masked_index_target = context.split().index(target_word) + 1
        input_sequence = self.tokenizer(context)
        attention_mask = torch.tensor(input_sequence["attention_mask"]).unsqueeze(0).to(self.device)
        saved_values = defaultdict(list)

        for i in tqdm(range(self.steps)):

            target_loss = None

            for i, word in enumerate(context.split()):
                input_ids = torch.tensor(input_sequence["input_ids"]).unsqueeze(0).to(self.device)
                masked_index = i + 1
                input_ids[0, masked_index] = self.tokenizer.mask_token_id
                target_id = torch.tensor(self.tokenizer.convert_tokens_to_ids([word])).to(self.device)[0]
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs[0][:, masked_index]
                loss = self.loss(logits, target_id.unsqueeze(0))
                saved_values[f"{word}_{masked_index}"].append(float(loss))
                if masked_index == masked_index_target:
                    target_loss = loss

            target_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        loss_for_words = pd.DataFrame.from_dict(saved_values)

        return loss_for_words


batch =[{"target_word": "apple", "context": "an apple is a fruit as wonderful as can be"},
        {"target_word": "apple", "context": "apple is a fruits"},
        {"target_word": "apple", "context": "fruits can be bananas , an apple , or others"},
        {"target_word": "apple", "context": "an apple is a great fruit"},
        {"target_word": "apple", "context": "you can eat an apple"},
        {"target_word": "apple", "context": "do you want an apple ?"},
        {"target_word": "apple", "context": "not only is an apple healthy , it is also tasty"},]

multiple_mask = [{"target_words": ["apple", "banana"], "context": "an apple is a fruit and so is a banana"}]

steps = 20
test = TestTargetLearning(model_name='bert-base-uncased', steps=steps)

loss_for_words_1 = test.learn_target_word_single(data=batch)
#loss_for_words_2 = test.learn_target_word_batch(helper=batch)
loss_for_words_3 = test.learn_target_words(data=multiple_mask)
comparison = pd.concat([loss_for_words_1, pd.Series([None]*steps), loss_for_words_3], axis=1)

s = 0
