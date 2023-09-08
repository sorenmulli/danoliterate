SRC_DIR := nlgenda
SRC_FILES := $(shell find $(SRC_DIR) -type f -name "*.py")

FORMATTERS := black isort
LINTERS := pylint mypy

MODELS_DIR := local-models

FASTTEXT_SOURCE := https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

.PHONY: all install format lint clean models-dir download-models fasttext

all: install

install:
	pip install --editable . --upgrade

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

download-models: models-dir fasttext
