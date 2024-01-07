from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from datasets import Dataset

from danoliterate.evaluation.execution.augmentation import Augmenter
from danoliterate.evaluation.execution.model_inference import ModelInference
from danoliterate.evaluation.results import ExecutionExample


class TaskRunner(ABC):
    can_build_multi_examples = False
    augmenter: Optional[Augmenter]

    def __init__(self, prompt_feature: str = "text", id_features: tuple[str, ...] = ("id",)):
        self.prompt_feature = prompt_feature
        # Allow ID to be concatenation of features
        self.id_features = id_features

    @abstractmethod
    def build_example(
        self,
        row: dict[str, Any],
        pre_prompt="",
        post_prompt="",
        idx: Optional[int] = None,
        few_shot_dataset: Optional[Dataset] = None,
    ) -> ExecutionExample:
        ...

    @abstractmethod
    def get_prediction(
        self, example: ExecutionExample, inference: ModelInference
    ) -> ExecutionExample:
        ...

    def get_example_id(self, row: dict[str, Any], idx: Optional[int] = None) -> str:
        if self.id_features:
            return "-".join(str(row[id_feature]) for id_feature in self.id_features)
        if idx is not None:
            return str(idx)
        raise ValueError("Neither ID features nor index where given; cannot identify row.")

    def prepare_prompt(self, row: dict[str, Any], pre_prompt: str, post_prompt: str) -> str:
        return pre_prompt + self.maybe_augment(row[self.prompt_feature]) + post_prompt

    def build_multi_examples(
        self,
        row: dict[str, Any],
        pre_prompt="",
        post_prompt="",
        idx: Optional[int] = None,
        few_shot_dataset: Optional[Dataset] = None,
    ) -> list[ExecutionExample]:
        raise NotImplementedError

    def set_augmenter(self, augmenter: Augmenter):
        self.augmenter = augmenter

    def maybe_augment(self, text: str) -> str:
        if self.augmenter is None:
            return text
        return self.augmenter(text)

    @property
    def is_few_shot(self):
        return False


class MultichoiceRunner(TaskRunner):
    def get_options(self, row: dict[str, Any]) -> list[str]:
        options = []
        for name, val in row.items():
            if "option" in name and isinstance(val, str):
                options.append(val)
        return options

    def get_correct_idx(self, row: dict[str, Any]) -> int:
        assert isinstance(row["correct"], int)
        return row["correct"]

    def build_example(
        self,
        row: dict[str, Any],
        pre_prompt="",
        post_prompt="",
        idx: Optional[int] = None,
        _: Optional[list[dict[str, Any]]] = None,
    ) -> ExecutionExample:
        return ExecutionExample(
            prompt=self.prepare_prompt(row, pre_prompt, post_prompt),
            id_=self.get_example_id(row, idx),
            options=self.get_options(row),
            index_label=self.get_correct_idx(row),
        )

    def get_prediction(
        self, example: ExecutionExample, inference: ModelInference
    ) -> ExecutionExample:
        assert example.options is not None

        if inference.can_do_lm:
            example.options_model_likelihoods = [
                inference.query_likelihood(example.prompt, option) for option in example.options
            ]
        if inference.can_do_nlg:
            example.generated_text = inference.query_generate_text(example.prompt)
        return example


class MultichoiceRunnerLetterOptions(MultichoiceRunner):
    """
    A multiple choice task that has the options saved as letters A, B, C, D instead of indices
    """

    def get_correct_idx(self, row: dict[str, Any]) -> int:
        num_options = sum("option" in key for key in row)
        # Generates the string ABCDE...
        letter_options = "".join(chr(i) for i in range(65, 65 + num_options))
        return letter_options.index(row["correct"])


class ClozeRunnerWithOptions(MultichoiceRunner):
    def __init__(
        self,
        prompt_feature: str = "text",
        id_features: tuple[str, ...] = ("id",),
        cloze_mask_key: str = "cloze",
        cloze_mask_replaced: str = "<MASK>",
    ):
        super().__init__(prompt_feature, id_features)
        self.cloze_mask_key = cloze_mask_key
        self.cloze_mask_replaced = cloze_mask_replaced

    def prepare_prompt(self, row: dict[str, Any], pre_prompt: str, post_prompt: str) -> str:
        return (
            pre_prompt
            + self.maybe_augment(
                row[self.prompt_feature].format(**{self.cloze_mask_key: self.cloze_mask_replaced})
            )
            + post_prompt.format(options=", ".join(self.get_options(row)))
        )


class MultichoiceRunnerWithOptions(MultichoiceRunner):
    def prepare_prompt(self, row: dict[str, Any], pre_prompt: str, post_prompt: str) -> str:
        options = "\n".join(f"{i+1}. {option}" for i, option in enumerate(self.get_options(row)))
        return (
            pre_prompt
            + self.maybe_augment(row[self.prompt_feature])
            + post_prompt.format(options=options)
        )

    def build_example(
        self,
        row: dict[str, Any],
        pre_prompt="",
        post_prompt="",
        idx: Optional[int] = None,
        _: Optional[list[dict[str, Any]]] = None,
    ) -> ExecutionExample:
        return ExecutionExample(
            prompt=self.prepare_prompt(row, pre_prompt, post_prompt),
            id_=self.get_example_id(row, idx),
            options=[str(i + 1) for i in range(len(self.get_options(row)))],
            index_label=self.get_correct_idx(row),
        )


class MultichoiceRunnerLetterWithOptions(
    MultichoiceRunnerWithOptions, MultichoiceRunnerLetterOptions
):
    ...


class MultichoiceRunnerLetterWithContext(MultichoiceRunnerLetterOptions):
    def __init__(
        self,
        prompt_feature: str = "text",
        id_features: tuple[str, ...] = ("id",),
        context_feature: str = "context",
    ):
        super().__init__(prompt_feature, id_features)
        self.context_feature = context_feature

    def prepare_prompt(self, row: dict[str, Any], pre_prompt: str, post_prompt: str) -> str:
        context = row[self.context_feature] or ""
        return (
            pre_prompt.format(context=context)
            + self.maybe_augment(row[self.prompt_feature])
            + post_prompt
        )


class MultichoiceRunnerLetterWithContextAndOptions(
    MultichoiceRunnerLetterWithContext, MultichoiceRunnerLetterWithOptions
):
    def prepare_prompt(self, row: dict[str, Any], pre_prompt: str, post_prompt: str) -> str:
        options = "\n".join(f"{i+1}. {option}" for i, option in enumerate(self.get_options(row)))
        context = row[self.context_feature] or ""
        return (
            pre_prompt.format(context=context)
            + self.maybe_augment(row[self.prompt_feature])
            + post_prompt.format(options=options)
        )


class MultiChoiceRunnerSameOptions(MultichoiceRunner):
    """
    A multiple choice task that has the a constant set of possible label options globally,
    across exaples
    """

    def __init__(
        self,
        all_labels: list[str],
        label_feature: str = "label",
        prompt_feature: str = "text",
        id_features: tuple[str, ...] = ("id",),
    ):
        super().__init__(prompt_feature=prompt_feature, id_features=id_features)
        self.label_feature = label_feature
        self.all_labels = [str(x) for x in all_labels]

    def get_correct_idx(self, row: dict[str, Any]) -> int:
        return self.all_labels.index(row[self.label_feature])

    def get_options(self, _: dict[str, Any]) -> list[str]:
        return self.all_labels


class AnswerSimilarityRunner(TaskRunner):
    def __init__(
        self,
        prompt_feature="prompt",
        answer_feature="answer",
        id_features: tuple[str, ...] = ("id",),
    ):
        super().__init__(prompt_feature, id_features)
        self.answer_feature = answer_feature

    def build_example(
        self,
        row: dict[str, Any],
        pre_prompt="",
        post_prompt="",
        idx: Optional[int] = None,
        _: Optional[list[dict[str, Any]]] = None,
    ) -> ExecutionExample:
        return ExecutionExample(
            prompt=self.prepare_prompt(row, pre_prompt, post_prompt),
            id_=self.get_example_id(row, idx),
            target_answer=row[self.answer_feature],
        )

    def get_prediction(
        self, example: ExecutionExample, inference: ModelInference
    ) -> ExecutionExample:
        assert example.target_answer is not None or example.options is not None

        if inference.can_do_nlg:
            example.generated_text = inference.query_generate_text(example.prompt)

        return example


class MultiAnswerSimilarityRunner(AnswerSimilarityRunner):
    def get_options(self, row: dict[str, Any]) -> list[str]:
        options = []
        for name, val in row.items():
            if self.answer_feature in name and isinstance(val, str):
                options.append(val)
        return options

    def build_example(
        self,
        row: dict[str, Any],
        pre_prompt="",
        post_prompt="",
        idx: Optional[int] = None,
        _: Optional[list[dict[str, Any]]] = None,
    ) -> ExecutionExample:
        return ExecutionExample(
            prompt=self.prepare_prompt(row, pre_prompt, post_prompt),
            id_=self.get_example_id(row, idx),
            options=self.get_options(row),
        )


class GptNerRunner(TaskRunner):
    can_build_multi_examples = True
    entity_start = "@@"
    entity_end = "##"

    def __init__(
        self,
        entity_types: list[dict[str, str]],
        prompt_feature="text",
        id_features: tuple[str, ...] = ("id",),
        token_feature="tokens",
        label_feature="labels",
        few_shot_format="Input: {text}\nOutput: {annotated_text}",
        num_examples=3,
    ):
        super().__init__(prompt_feature, id_features)
        self.entity_type_map = {key: value for dct in entity_types for key, value in dct.items()}
        self.prompt_feature = prompt_feature
        self.num_examples = num_examples
        self.token_feature = token_feature
        self.label_feature = label_feature
        self.few_shot_format = few_shot_format
        self.id_features = id_features
        self.rng = np.random.default_rng(seed=1887)

    def annotate_ground_truth(self, example: dict[str, Any], entity: str) -> str:
        words = []
        is_open_entity = False
        for token, label in zip(
            example[self.token_feature], example[self.label_feature], strict=True
        ):
            # If we found an entity and none was open, start a new
            if label.split("-")[-1] == entity:
                if not is_open_entity:
                    words.append(self.entity_start + token)
                    is_open_entity = True
                    continue
            # If no entity and one was open, close the last one
            elif is_open_entity:
                words[-1] = words[-1] + self.entity_end
                is_open_entity = False
            words.append(token)
        if is_open_entity:
            words[-1] = words[-1] + self.entity_end
        return " ".join(words)

    def build_ner_prompt(
        self,
        text: str,
        few_shot_examples: list[dict[str, str]],
        pre_prompt: str,
        post_prompt: str,
        entity: str,
    ) -> str:
        few_shot_str = "\n".join(
            self.few_shot_format.format(
                text=example[self.prompt_feature],
                annotated_text=self.annotate_ground_truth(example, entity),
            )
            for example in few_shot_examples
        )
        return (
            pre_prompt.format(entity_str=self.entity_type_map[entity], few_shot_str=few_shot_str)
            + text
            + post_prompt
        )

    def _sample_few_shot(self, dataset: Dataset, entity: str) -> list[tuple[int, dict[str, Any]]]:
        shuffled_idcs = self.rng.choice(len(dataset), len(dataset), replace=False)
        labels = dataset[self.label_feature]
        i = 0
        for i, idx in enumerate(shuffled_idcs):
            if entity in {entity.split("-")[-1] for entity in labels[idx] if "-" in entity}:
                break
        # Index backwards as to always get num examples
        chosen = shuffled_idcs[i : i + self.num_examples]
        assert len(chosen) == self.num_examples
        return [(int(i), ex) for (i, ex) in zip(chosen, dataset.select(chosen), strict=True)]

    def build_multi_examples(
        self,
        row: dict[str, Any],
        pre_prompt="",
        post_prompt="",
        idx: Optional[int] = None,
        few_shot_dataset: Optional[Dataset] = None,
    ) -> list[ExecutionExample]:
        assert few_shot_dataset is not None
        examples = []
        id_ = self.get_example_id(row, idx)
        for entity in self.entity_type_map:
            few_shot_examples = self._sample_few_shot(few_shot_dataset, entity)
            examples.append(
                ExecutionExample(
                    prompt=self.build_ner_prompt(
                        self.maybe_augment(row[self.prompt_feature]),
                        [ex for _, ex in few_shot_examples],
                        pre_prompt,
                        post_prompt,
                        entity,
                    ),
                    id_=id_ + f"-{entity}",
                    few_shot_example_ids=[
                        self.get_example_id(example, i) for i, example in few_shot_examples
                    ],
                )
            )
        return examples

    def get_prediction(
        self, example: ExecutionExample, inference: ModelInference
    ) -> ExecutionExample:
        assert inference.can_do_nlg
        query = inference.query_generate_text(example.prompt)
        example.generated_text = query
        example.generated_score = query.query_score()
        return example

    @property
    def is_few_shot(self):
        return True

    def build_example(
        self,
        row: dict[str, Any],
        pre_prompt="",
        post_prompt="",
        idx: Optional[int] = None,
        few_shot_dataset: Optional[Dataset] = None,
    ) -> ExecutionExample:
        raise NotImplementedError
