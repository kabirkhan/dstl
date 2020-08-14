#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place docs/src/ dstl tests --exclude=__init__.py
black dstl tests docs/src
isort dstl tests docs/src