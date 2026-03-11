# Autodocumented Makefile for xDEM
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# Dependencies : python3 venv

############### GLOBAL VARIABLES ######################
.DEFAULT_GOAL := help
SHELL := /bin/bash

# Virtualenv directory name (can be overridden)
ifndef VENV
	VENV = "venv"
endif

# Python global variables definition
PYTHON_VERSION_MIN = 3.10
# Set PYTHON if not defined in command line
# Example: PYTHON="python3.13" make venv to use python 3.13 for the venv
# By default the default python3 of the system.
ifndef PYTHON
	PYTHON = "python3"
endif
PYTHON_CMD=$(shell command -v $(PYTHON))

PYTHON_VERSION_CUR=$(shell $(PYTHON_CMD) -c 'import sys; print("%d.%d"% sys.version_info[0:2])')
PYTHON_VERSION_OK=$(shell $(PYTHON_CMD) -c 'import sys; cur_ver = sys.version_info[0:2]; min_ver = tuple(map(int, "$(PYTHON_VERSION_MIN)".split("."))); print(int(cur_ver >= min_ver))')

############### Check python version supported ############

ifeq (, $(PYTHON_CMD))
    $(error "PYTHON_CMD=$(PYTHON_CMD) not found in $(PATH)")
endif

ifeq ($(PYTHON_VERSION_OK), 0)
    $(error "Requires Python version >= $(PYTHON_VERSION_MIN). Current version is $(PYTHON_VERSION_CUR)")
endif

################ MAKE Targets ######################

help: ## Show this help
	@echo "      XDEM MAKE HELP"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: venv
venv: ## Create a virtual environment in 'venv' directory if it doesn't exist
	@test -d ${VENV} || $(PYTHON_CMD) -m venv ${VENV}
	@touch ${VENV}/bin/activate
	@${VENV}/bin/python -m pip install --upgrade wheel setuptools pip


.PHONY: install
install: venv ## Install xDEM for development (depends on venv)
	@test -f ${VENV}/bin/xdem || echo "Installing xdem in development mode"
	@test -f ${VENV}/bin/xdem || ${VENV}/bin/pip install -e .[dev]
	@test -f .git/hooks/pre-commit || echo "Installing pre-commit hooks"
	@test -f .git/hooks/pre-commit || ${VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${VENV}/bin/pre-commit install -t pre-push
	@echo "xDEM installed in development mode in virtualenv ${VENV}"
	@echo "To use: source ${VENV}/bin/activate; xdem -h"


.PHONY: tests
tests: ## run tests
	@${VENV}/bin/pytest

## Clean section

.PHONY: clean
clean: clean-venv clean-build clean-pyc clean-precommit ## Clean all

.PHONY: clean-venv
clean-venv: ## Clean the virtual environment
	@echo "+ $@"
	@rm -rf ${VENV}

.PHONY: clean-build
clean-build: ## Remove build artifacts
	@echo "+ $@"
	@rm -rf build/ dist/ .eggs/
	@find . -name '*.egg-info' -exec rm -rf {} +
	@find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-precommit
clean-precommit: ## Remove pre-commit hooks from .git/hooks
	@rm -f .git/hooks/pre-commit
	@rm -f .git/hooks/pre-push

.PHONY: clean-pyc
clean-pyc: ## Remove Python cache and artifacts
	@echo "+ $@"
	@find . -type f -name "*.py[co]" -exec rm -rf {} +
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -name '*~' -exec rm -rf {} +
