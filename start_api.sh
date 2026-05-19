#!/bin/bash
cd "$(dirname "$0")"
source voxcpm-env/bin/activate
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg/lib
export MODEL_PATH=models/openbmb__VoxCPM2
export LOAD_DENOISER=false
exec python api_server.py
