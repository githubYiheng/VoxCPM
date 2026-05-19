#!/bin/bash
cd "$(dirname "$0")"
source voxcpm-env/bin/activate
DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg/lib python app.py
