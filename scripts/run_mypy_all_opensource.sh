#!/bin/sh

cd "$(dirname "$0")"
../core/run_mypy.sh
../genotypes/cppnwin/run_mypy.sh