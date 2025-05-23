#!/bin/bash

models=(VTTSAT VTTPAT)

### 100_ref
for model in ${models[@]}
do
  python main.py \
  --train \
  --model $model \
  --dataname GDN_step45_100_ref  \
  --configure config.yaml
done

### 100_ref_val
for model in ${models[@]}
do
  python main.py \
  --train \
  --model $model \
  --dataname GDN_step45_100_ref_val \
  --configure config_val.yaml
done

python main.py --train --model VTTPAT --dataname GDN_step45_big_ref --configure config.yaml

python main.py --train --model VTTPAT --dataname GDN_step45_big_ref_val --configure config_val.yaml




# ### SWaT
# for model in ${models[@]}
# do
#   python main.py \
#   --train \
#   --test \
#   --model $model \
#   --dataname SWaT \
#   --use_multi_gpu \
#   --devices 0,1,2,3
# done
