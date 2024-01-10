import re

import torch
from danlp.models import BertOffensive


class BatchBertOffensive(BertOffensive):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def _get_batch_pred(self, sentences: list[str]) -> torch.Tensor:
        inputs = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.device)
        with torch.inference_mode():
            pred = self.model(**inputs)[0]
        return pred

    def batch_predict_proba(self, sentences: list[str]) -> list[float]:
        all_probas = []
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i : i + self.batch_size]
            batch_sentences = _preprocess_punctuation(batch_sentences)
            preds = self._get_batch_pred(batch_sentences)
            probas = torch.nn.functional.softmax(preds, dim=1).detach().cpu().numpy()[:, 1]
            all_probas.extend(probas)
        return [float(prob) for prob in all_probas]


def _preprocess_punctuation(sentences: list[str]) -> list[str]:
    # Collapse many following punctuation marks give very high offensive
    # probabilitiesl
    return [re.sub(r"([.,!?;:])\1{3,}", lambda m: m.group(0)[:3], text) for text in sentences]
