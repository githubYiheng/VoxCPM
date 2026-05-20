#!/bin/bash
cd "$(dirname "$0")"
source voxcpm-env/bin/activate

# macOS: torchcodec needs ffmpeg libs on the loader path.
# Use brew's version-independent prefix so this works across machines/ffmpeg versions.
if command -v brew >/dev/null 2>&1; then
  export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib:$DYLD_LIBRARY_PATH"
fi

python app.py --model-id models/VoxCPM2 --port 7860
