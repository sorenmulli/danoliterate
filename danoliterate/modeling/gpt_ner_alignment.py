import re

from Levenshtein import editops
from nltk import word_tokenize


def default_ner_tokenize(text):
    tokens = word_tokenize(text)
    custom_tokens = []
    for token in tokens:
        # NLTK word tokenize does not exactly match DaNE
        if re.match(r".+\'\w+", token):
            custom_tokens.extend(part for part in re.split(r"(\'\w+)", token) if part)
        else:
            custom_tokens.append(token)
    fixed_tokens = []
    i = 0
    while i < len(custom_tokens):
        if i + 2 < len(custom_tokens) and custom_tokens[i] == "@" and custom_tokens[i + 1] == "@":
            fixed_tokens.append("@@" + custom_tokens[i + 2])
            i += 3
        elif (
            i - 1 >= 0 and custom_tokens[i] == "#" and custom_tokens[i - 1] == "#" and fixed_tokens
        ):
            fixed_tokens[-1] += "##"
            i += 1
        else:
            if custom_tokens[i] not in ["@", "#"]:
                fixed_tokens.append(custom_tokens[i])
            i += 1

    return fixed_tokens


def clean_annotations(text: str) -> str:
    annotations = re.findall(r"@@|##", text)
    open_annotation = False
    for annotation in annotations:
        if annotation == "@@":
            if open_annotation:
                text = text.replace("@@", "", 1)
            else:
                open_annotation = True
        elif annotation == "##":
            if open_annotation:
                open_annotation = False
            else:
                text = text.replace("##", "", 1)
    if open_annotation:
        text = text.replace("@@", "")
    return text


def parse_model_pred(tokens: list[str], generated: str, entity: str) -> list[str]:
    generated_tokens = default_ner_tokenize(generated)
    comparison_gen_tokens = [
        token.replace("@@", "").replace("##", "") for token in generated_tokens
    ]
    model_prediction = ["O"] * len(generated_tokens)
    aligned_tokens = generated_tokens.copy()
    # Run levenshtein alignment
    offset = 0
    for opcode, source_pos, _ in editops(comparison_gen_tokens, tokens):
        match opcode:
            case "delete":
                del model_prediction[source_pos + offset]
                del aligned_tokens[source_pos + offset]
                offset -= 1
            case "insert":
                model_prediction.insert(source_pos + offset, "O")
                aligned_tokens.insert(source_pos + offset, "")
                offset += 1
    in_entity = False
    for i, token in enumerate(aligned_tokens):
        if token.startswith("@@"):
            model_prediction[i] = "B-" + entity
            if not token.endswith("##"):
                in_entity = True
        elif in_entity and token.endswith("##"):
            model_prediction[i] = "I-" + entity
            in_entity = False
        elif in_entity:
            model_prediction[i] = "I-" + entity
    # from pelutils import Table
    # table = Table()
    # table.add_row(tokens)
    # table.add_row(aligned_tokens)
    # table.add_row(model_prediction)
    # print(table)
    assert len(model_prediction) == len(tokens)
    return model_prediction
