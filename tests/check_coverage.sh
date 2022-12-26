#!/usr/bin/env bash
# File       : check_coverage.sh
# Description: Coverage wrapper around test suite driver script
# Copyright 2022 Harvard University. All Rights Reserved.
set -e

# Must add the module source path because we use `import autodiff30` in
# our test suite.  This is necessary if you want to test in your local
# development environment without properly installing the package.
export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}

# run the coverage reort
#pytest --cov=autodiff30 --cov-report=term-missing > coverage_report.txt
./run_tests.sh pytest --cov=autodiff30 "${@}"
