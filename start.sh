#!/bin/bash
cd "$(dirname "$0")"
source voxcpm-env/bin/activate
DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/ffmpeg/7.1.1_3/lib python app.py
