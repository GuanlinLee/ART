#!/bin/bash

SD_VERSION="1.5"
EXP_FOLDER="test"
GUIDANCE="7.5"
HEIGHT=512
WIDTH=512
SAVE_FOLDER="test_gen"

python generate_images.py \
--sd_version $SD_VERSION \
--exp_folder $EXP_FOLDER \
--guidance_scale $GUIDANCE \
--height $HEIGHT \
--width $WIDTH \
--save_folder $SAVE_FOLDER
