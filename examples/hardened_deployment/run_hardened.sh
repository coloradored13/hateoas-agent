#!/usr/bin/env bash
# Launch the hateoas-agent gate as a locked-down process that holds the SOLE
# resource credential. The agent (your MCP client) is launched separately,
# WITHOUT RESOURCE_DB_DSN in its environment, and reaches the resource only by
# sending tool calls to this gate over MCP stdio.
#
# The credential split is the whole point: because the agent process has no
# resource credential and no direct path, the state gate becomes a real
# boundary rather than an advisory contract. See ../../SECURITY.md.
set -euo pipefail

# The resource credential exists ONLY in this gate process's environment.
export RESOURCE_DB_DSN="${RESOURCE_DB_DSN:-postgres://gate@localhost/orders}"

# In production, run as a dedicated unprivileged user that owns the credential,
# e.g.:  exec sudo -u gate --preserve-env=RESOURCE_DB_DSN python3 server.py
# Do NOT export RESOURCE_DB_DSN into the agent process's environment.
exec python3 "$(dirname "$0")/server.py"
