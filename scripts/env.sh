#!/usr/bin/env bash
# Shared env for MiniOneRec (mor conda). Prefer this over scripts/env_mor.sh.
# shellcheck source=/dev/null
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep historical name working
source "$ROOT/scripts/env_mor.sh"
