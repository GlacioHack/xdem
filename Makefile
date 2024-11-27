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

# Python version requirement
PYTHON_VERSION_REQUIRED = 3.10

ifndef PYTHON
	# Try to find python version required
	PYTHON = "python$(PYTHON_VERSION_REQUIRED)"
endif
PYTHON_CMD=$(shell command -v $(PYTHON))

PYTHON_VERSION_CUR=$(shell $(PYTHON_CMD) -c 'import sys; print("%d.%d" % sys.version_info[0:2])')
PYTHON_VERSION_OK=$(shell $(PYTHON_CMD) -c 'import sys; req_ver = tuple(map(int, "$(PYTHON_VERSION_REQUIRED)".split("."))); cur_ver = sys.version_info[0:2]; print(int(cur_ver == req_ver))')

############### Check python version supported ############

ifeq (, $(PYTHON_CMD))
    $(error "PYTHON_CMD=$(PYTHON_CMD) not found in $(PATH)")
endif

ifeq ($(PYTHON_VERSION_OK), 0)
    $(error "Requires Python version == $(PYTHON_VERSION_REQUIRED). Current version is $(PYTHON_VERSION_CUR)")
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

.PHONY: install-gdal
install-gdal: ## Install GDAL version matching the system's GDAL via pip
	@if command -v gdalinfo >/dev/null 2>&1; then \
		GDAL_VERSION=$$(gdalinfo --version | awk '{print $$2}'); \
		echo "System GDAL version: $$GDAL_VERSION"; \
		${VENV}/bin/pip install gdal==$$GDAL_VERSION; \
	else \
		echo "Warning: GDAL not found on the system. Proceeding without GDAL."; \
		echo "Try installing GDAL by running the following commands depending on your system:"; \
		echo "Debian/Ubuntu: sudo apt-get install -y gdal-bin libgdal-dev"; \
		echo "Red Hat/CentOS: sudo yum install -y gdal gdal-devel"; \
		echo "Then run 'make install-gdal' to proceed with GDAL installation."; \
	fi

.PHONY: install
install: venv ## Install xDEM for development (depends on venv)
	@test -f ${VENV}/bin/xdem || echo "Installing xdem in development mode"
	@test -f ${VENV}/bin/xdem || ${VENV}/bin/pip install -e .[dev]
	@test -f .git/hooks/pre-commit || echo "Installing pre-commit hooks"
	@test -f .git/hooks/pre-commit || ${VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${VENV}/bin/pre-commit install -t pre-push
	@echo "Attempting to install GDAL..."
	@make install-gdal
	@echo "xdem installed in development mode in virtualenv ${VENV}"
	@echo "To use: source ${VENV}/bin/activate; xdem -h"


.PHONY: test
test: ## run tests
	@if ! ${VENV}/bin/python -m pip show gdal >/dev/null 2>&1; then \
		echo "Error: GDAL is not installed in the virtual environment. Tests require GDAL to run."; \
		echo "Please ensure GDAL is installed by running 'make install-gdal'."; \
		exit 1; \
	else \
		${VENV}/bin/pytest; \
	fi

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
