from statistics import mean

from nlgenda.evaluation.example import EvaluationExample, MultichoiceExample

# TODO: Where should the metric calculation logic be?


class Evaluator:
    def __init__(self, result_db: str):
        self.results: list[int] = []
        self.result_db = result_db

    def receive_result(self, example: EvaluationExample):
        if isinstance(example, MultichoiceExample):
            assert example.prediction is not None
            self.results.append(1 if example.prediction == example.label else 0)
        else:
            raise NotImplementedError

    def finish(self):
        print(mean(self.results) * 100)
