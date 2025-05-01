# Check if pixi is available, if not use ssmuse to get it
PIXI_CHECK := $(shell command -v pixi 2> /dev/null)
PIXI_CMD := $(if $(PIXI_CHECK),pixi,. ssmuse-sh -p /fs/ssm/eccc/cmd/cmds/apps/pixi/202504/00/pixi_0.44.0_all && pixi)

.PHONY: test lint lint-fix conda-build conda-upload run-both run-py38 run-py313 clean

# Development targets
test:
	@echo "********* Running test target *********"
	$(PIXI_CMD) run -e dev test

lint:
	@echo "********* Running lint target *********"
	$(PIXI_CMD) run -e dev lint

lint-fix:
	@echo "********* Running lint-fix target *********"
	$(PIXI_CMD) run -e dev lint-fix

format:
	@echo "********* Running format target *********"
	$(PIXI_CMD) run -e dev format

# Conda package management
conda-build:
	@echo "********* Running conda-build target *********"
	$(PIXI_CMD) run -e dev conda-build

conda-upload:
	@echo "********* Running conda-upload target *********"
	$(PIXI_CMD) run -e dev conda-upload

test-py38: clean
	@echo "********* Testing package with python 3.8 *********"
	cd package_tests && $(PIXI_CMD) run -e py38 test

test-py39: clean
	@echo "********* Testing package with python 3.9 *********"
	cd package_tests && $(PIXI_CMD) run -e py39 test

test-py310: clean
	@echo "********* Testing package with python 3.10 *********"
	cd package_tests && $(PIXI_CMD) run -e py310 test

test-py311: clean
	@echo "********* Testing package with python 3.11 *********"
	cd package_tests && $(PIXI_CMD) run -e py311 test

test-py312: clean
	@echo "********* Testing package with python 3.12 *********"
	cd package_tests && $(PIXI_CMD) run -e py312 test

test-py313: clean
	@echo "********* Testing package with python 3.13 *********"
	cd package_tests && $(PIXI_CMD) run -e py313 test

test-all: test-py38 test-py39 test-py310 test-py311 test-py312 test-py313

# Clean target (if needed)
clean:
	$(PIXI_CMD) clean
	cd package_tests && $(PIXI_CMD) clean
