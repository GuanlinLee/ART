#!/bin/bash

# path to the llava lora
LLAVA_LORA_PATH="glli/ART_GuideModel" # path to the llava lora
LLAVA_MODEL_PATH="liuhaotian/llava-v1.6-mistral-7b"

# path to the llama lora
LLAMA_LORA_PATH="glli/ART_WriterModel" # path to the llama lora
LLAMA_MODEL_PATH="meta-llama/Llama-2-7b-hf" # path to the llama 2 7b model

SD_VERSION="1.5"
SEED=1234
EXP_FOLDER="test"
CATEGORY="sexual"
ROUND=50
GUIDANCE="7.5"
HEIGHT=512
WIDTH=512


python run_agents.py --llama_model_path $LLAMA_MODEL_PATH \
        --llama_lora_path $LLAMA_LORA_PATH \
        --sd_version $SD_VERSION \
        --llava_model_path $LLAVA_MODEL_PATH \
        --llava_lora_path $LLAVA_LORA_PATH \
        --seed $SEED \
        --exp_folder $EXP_FOLDER \
        --concept $CATEGORY \
        --round $ROUND \
        --guidance_scale $GUIDANCE \
        --height $HEIGHT \
        --width $WIDTH


CATEGORY="hate"

python run_agents.py --llama_model_path $LLAMA_MODEL_PATH \
        --llama_lora_path $LLAMA_LORA_PATH \
        --sd_version $SD_VERSION \
        --llava_model_path $LLAVA_MODEL_PATH \
        --llava_lora_path $LLAVA_LORA_PATH \
        --seed $SEED \
        --exp_folder $EXP_FOLDER \
        --concept $CATEGORY \
        --round $ROUND \
        --guidance_scale $GUIDANCE \
        --height $HEIGHT \
        --width $WIDTH

CATEGORY="harassment"

python run_agents.py --llama_model_path $LLAMA_MODEL_PATH \
        --llama_lora_path $LLAMA_LORA_PATH \
        --sd_version $SD_VERSION \
        --llava_model_path $LLAVA_MODEL_PATH \
        --llava_lora_path $LLAVA_LORA_PATH \
        --seed $SEED \
        --exp_folder $EXP_FOLDER \
        --concept $CATEGORY \
        --round $ROUND \
        --guidance_scale $GUIDANCE \
        --height $HEIGHT \
        --width $WIDTH

CATEGORY="violence"

python run_agents.py --llama_model_path $LLAMA_MODEL_PATH \
        --llama_lora_path $LLAMA_LORA_PATH \
        --sd_version $SD_VERSION \
        --llava_model_path $LLAVA_MODEL_PATH \
        --llava_lora_path $LLAVA_LORA_PATH \
        --seed $SEED \
        --exp_folder $EXP_FOLDER \
        --concept $CATEGORY \
        --round $ROUND \
        --guidance_scale $GUIDANCE \
        --height $HEIGHT \
        --width $WIDTH

CATEGORY="self-harm"

python run_agents.py --llama_model_path $LLAMA_MODEL_PATH \
        --llama_lora_path $LLAMA_LORA_PATH \
        --sd_version $SD_VERSION \
        --llava_model_path $LLAVA_MODEL_PATH \
        --llava_lora_path $LLAVA_LORA_PATH \
        --seed $SEED \
        --exp_folder $EXP_FOLDER \
        --concept $CATEGORY \
        --round $ROUND \
        --guidance_scale $GUIDANCE \
        --height $HEIGHT \
        --width $WIDTH

CATEGORY="shocking"

python run_agents.py --llama_model_path $LLAMA_MODEL_PATH \
        --llama_lora_path $LLAMA_LORA_PATH \
        --sd_version $SD_VERSION \
        --llava_model_path $LLAVA_MODEL_PATH \
        --llava_lora_path $LLAVA_LORA_PATH \
        --seed $SEED \
        --exp_folder $EXP_FOLDER \
        --concept $CATEGORY \
        --round $ROUND \
        --guidance_scale $GUIDANCE \
        --height $HEIGHT \
        --width $WIDTH

CATEGORY="illegal_activity"

python run_agents.py --llama_model_path $LLAMA_MODEL_PATH \
        --llama_lora_path $LLAMA_LORA_PATH \
        --sd_version $SD_VERSION \
        --llava_model_path $LLAVA_MODEL_PATH \
        --llava_lora_path $LLAVA_LORA_PATH \
        --seed $SEED \
        --exp_folder $EXP_FOLDER \
        --concept $CATEGORY \
        --round $ROUND \
        --guidance_scale $GUIDANCE \
        --height $HEIGHT \
        --width $WIDTH