import torch
from danlp.models import BertOffensive


class BatchBertOffensive(BertOffensive):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def _get_batch_pred(self, sentences: list[str]) -> torch.Tensor:
        inputs = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        with torch.inference_mode():
            pred = self.model(inputs["input_ids"], token_type_ids=inputs["token_type_ids"])[0]
        return pred

    def batch_predict_proba(self, sentences: list[str]) -> list[float]:
        all_probas = []
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i : i + self.batch_size]
            preds = self._get_batch_pred(batch_sentences)
            probas = torch.nn.functional.softmax(preds, dim=1).detach().numpy()
            all_probas.extend(probas)

        return [prob.tolist() for prob in all_probas]
