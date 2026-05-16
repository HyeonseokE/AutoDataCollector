"""gRPC transport for the Method 3 preselective acquirer.

server.py    : H100-side service (curobo + buffer-only Selector + Buffer)
client.py    : robot-edge stub — drop-in replacement for the local hook
preselective.proto : IDL (regenerate stubs with grpcio-tools)
"""
from __future__ import annotations
