SRC_DIR := nlgenda
SRC_FILES := $(shell find $(SRC_DIR) -type f -name "*.py")

FORMATTERS := black isort
LINTERS := pylint mypy

MODELS_DIR := local-models

FASTTEXT_SOURCE := https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
DSL3GRAM_SOURCE := https://github.com/danspeech/danspeech/releases/download/v0.02-alpha/dsl_3gram.klm

.PHONY: all install format lint clean models-dir download-models fasttext dsl3gram

all: install

install:
	pip install --editable . --upgrade
	python -c 'import nltk; nltk.download("punkt")'
	python -m spacy download da_core_news_sm

tidy: clean format lint

format:
	@for formatter in $(FORMATTERS); do \
		echo 🧹 $$formatter && \
		$$formatter $(SRC_FILES); \
	done

lint:
	@for linter in $(LINTERS); do \
		echo 🔍 $$linter && \
		$$linter $(SRC_FILES); \
	done

clean:
	@find . -name "*.pyc" -exec rm -f {} +
	@find . -name "__pycache__" -type d -exec rm -rf {} +

models-dir:
	@mkdir -p $(MODELS_DIR)

fasttext:
	@wget $(FASTTEXT_SOURCE) -P $(MODELS_DIR)

dsl3gram:
	@wget $(DSL3GRAM_SOURCE) -P $(MODELS_DIR)

download-models: models-dir fasttext dsl3gram
