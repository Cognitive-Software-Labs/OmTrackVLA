#!/usr/bin/env bash
# Avoid killing unrelated evaluation jobs by process-name matching.
echo 'Stop your evaluation with Ctrl-C in its eval.sh terminal. Only that launcher owns its child process.' >&2
exit 1
