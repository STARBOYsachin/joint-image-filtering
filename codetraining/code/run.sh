#!/bin/bash
set -ex
# check to see if all required arguments were provided
if [ $# -eq 10 ]; then
    # assign the provided arguments to variables
    Task=$1
    model_type=$2
    image_type=$3
    image_path=$4
    image_url=$5
    gpu=$6
    r=$7
    eps=$8
    low_size=$9
    optional=${10}
    python -u main.py "$Task" "$model_type" "$image_type" "$image_path" "$image_url" "$gpu" "$r" \
      "$eps" "$low_size" "$optional" 
    
else
    # assign the default values to variables
    for i in auto_ps style_transfer; do
      Task="$i"
      model_type="deep_guided_filter"
      image_type="FILE"
      image_path="../data/images/$i.jpg"
      image_url="http://www.technocrazed.com/wp-content/uploads/2015/12/Landscape-wallpaper-7.jpg"
      gpu="0"
      r="-1"
      eps="-1"
      low_size="64"
      optional="--post_sigmoid --dgf --dgf_r 8 --dgf_eps 1e-2 --thres 161"
      python -u main.py "$Task" "$model_type" "$image_type" "$image_path" "$image_url" "$gpu" "$r" \
        "$eps" "$low_size" "$optional" 
      mkdir ../results/$i && mv ../results/*.jpg ../results/$i
    done

fi
