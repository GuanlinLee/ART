#!/bin/bash

SD_VERSION="1.5"
HEIGHT=512
WIDTH=512
SAVE_FOLDER="test_gen"
GUIDANCE="7.5"

python summarize_results.py \
--sd_version $SD_VERSION \
--guidance_scale $GUIDANCE \
--height $HEIGHT \
--width $WIDTH \
--save_folder $SAVE_FOLDER
