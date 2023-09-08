SRC_DIR := nlgenda
SRC_FILES := $(shell find $(SRC_DIR) -type f -name "*.py")

FORMATTERS := black isort
LINTERS := pylint mypy

.PHONY: all install format lint clean

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
