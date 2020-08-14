#!/usr/bin/env bash

set -e
set -x

mypy dstl --disallow-untyped-defs
black dstl tests --check
isort dstl tests docs/src --check-only
