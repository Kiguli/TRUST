#!/usr/bin/env bash
set -euo pipefail
bash -n scripts/deploy.sh
shellcheck scripts/deploy.sh
python3 tests/deployment_test.py
