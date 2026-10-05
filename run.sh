#!/usr/bin/env bash

set -euo pipefail

project_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
executable="$project_directory/bin/cuda_stream_dnn"

if [[ -f "$executable.exe" ]]; then
  executable="$executable.exe"
fi

if [[ ! -x "$executable" && ! -f "$executable" ]]; then
  echo "Executable not found: $executable" >&2
  echo "Build the project first. See INSTALL for the cuDNN setup command." >&2
  exit 1
fi

mkdir -p "$project_directory/data/results"
cd "$project_directory"
"$executable" "$@"
