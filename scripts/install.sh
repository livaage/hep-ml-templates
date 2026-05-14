#!/usr/bin/env bash
# Install hep-ml-templates with a chosen extras bundle.
#
# Usage:
#   scripts/install.sh                          # core only
#   scripts/install.sh xgb                      # one extra
#   scripts/install.sh pipeline-gnn             # full pipeline bundle
#   scripts/install.sh xgb data-higgs           # multiple extras
#   scripts/install.sh all                      # everything
#
# The extras listed here must match pyproject.toml [project.optional-dependencies].

set -euo pipefail

if [[ $# -eq 0 ]]; then
  EXTRAS="core"
else
  EXTRAS=$(IFS=, ; echo "$*")
fi

cd "$(dirname "$0")/.."
pip install -e ".[${EXTRAS}]"
