#!/bin/bash
# WS1 카메라 연결 확인
cd "$(dirname "${BASH_SOURCE[0]}")"
python check_cameras.py ws1
