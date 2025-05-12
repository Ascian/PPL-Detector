from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score
)
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


class PPLPredictor:
    def __init__(self, model_name, prompt_max_seq_len, max_seq_len, ppl_limit, ppl_threshold, ppl_scale, device='cuda'):
        self.prompt_max_seq_len = prompt_max_seq_len
        self.max_seq_len = max_seq_len
        self.ppl_limit = ppl_limit
        self.ppl_threshold = ppl_threshold
        self.ppl_scale = ppl_scale
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def evaluate(self, val_dataset):
        all_scores = []
        all_predictions = []
        all_true_labels = []

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(val_dataset, desc="Validation"):
                score = self.compute_score(sample['prompt'], sample['text'])
                prediction = int(score > 0.5)

                all_predictions.append(prediction)
                all_scores.append(score)
                all_true_labels.append(sample['label'])

        # Calculate AUC
        auc = roc_auc_score(all_true_labels, all_scores)
        
        # Calculate F1 score
        f1 = f1_score(all_true_labels, all_predictions)
        
        # Calculate precision and recall
        precision = precision_score(all_true_labels, all_predictions)
        recall = recall_score(all_true_labels, all_predictions)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_true_labels, all_predictions).ravel()
        
        # Calculate accuracy
        accuracy = accuracy_score(all_true_labels, all_predictions)  # Convert to percentage
        
        # Store results
        result = {
            "auc": auc,
            "accuracy": accuracy,  # Acc.(%)
            "f1": f1,             # F1
            "tn": tn,             # TN
            "tp": tp,             # TP
            "fn": fn,             # FN
            "fp": fp,             # FP
            "precision": precision, # Prec
            "recall": recall,     # Rec
            "weighted_score": 0.6 * auc + 0.3 * accuracy + 0.1 * f1, # Weighted Score
        }
        
        return result

    def predict(self, test_dataset):
        all_prompts = []
        all_scores = []
        
        for sample in tqdm(test_dataset, desc="Prediction"):
            score = self.compute_score(sample['prompt'], sample['text'])

            all_prompts.append(sample['prompt'])
            all_scores.append(score)

        return all_prompts, all_scores

    def compute_score(self, prompt, text):
        token_loss = self._calculate_token_loss(prompt, text)
        token_loss = token_loss[token_loss > 0]
        token_ppl = np.exp(token_loss)
        token_ppl[token_ppl > self.ppl_limit] = self.ppl_limit
        avg_ppl = np.mean(token_ppl)
        score = 1 / (1 + np.exp(-(avg_ppl - self.ppl_threshold) / self.ppl_scale))
        return score

    def _calculate_token_loss(self,  prompt, text):
        input,  loss_mask = self._tokenize(prompt, text)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input.unsqueeze(0),
                labels=input.clone()
            )
            logits = outputs.logits[0]
            shift_logits = logits[:-1, :].contiguous()
            shift_labels = input[1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_loss = loss_fct(shift_logits, shift_labels)

            token_loss = token_loss * loss_mask[1:] # remove the first token
            token_loss = token_loss.cpu().numpy() # batch_size x seq_len - 1

        return  token_loss

    def _tokenize(self, prompt, text):
        prompt_tokens = self.tokenizer(
            prompt, 
            return_tensors='pt'
        ).input_ids.squeeze(0).to(self.device)
        if len(prompt_tokens) > self.prompt_max_seq_len:
            prompt_tokens = prompt_tokens[-self.prompt_max_seq_len:]

        text_tokens = self.tokenizer(
            text, 
            max_length=self.max_seq_len - len(prompt_tokens), 
            truncation=True, 
            return_tensors='pt'
        ).input_ids.squeeze(0).to(self.device)

        if len(prompt_tokens) == 0:
            encoding = text_tokens
        else:
            encoding = torch.cat([prompt_tokens, text_tokens])

        prompt_lens_tensor = torch.tensor(len(prompt_tokens), device=self.device)
        indices = torch.arange(len(encoding), device=self.device)
        loss_mask = indices >= prompt_lens_tensor 

        return encoding,  loss_mask