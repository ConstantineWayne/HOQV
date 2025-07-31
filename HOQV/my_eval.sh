#!/bin/bash
for i in {1..5}
do
  for noise in {5,10}
  do
    for seed in {1..5}
    do
      task="MVSA_Single"
      task_type="classification"
      model="HOQV"
      name=$task"_"$model"_model_run_qmf_$i"
      echo $name noise: $noise seed: $seed
      CUDA_VISIBLE_DEVICES=4 python my_eval.py --batch_sz 16 --gradient_accumulation_steps 40  \
      --savedir ./saved/$task --name $name   \
       --task $task --task_type $task_type  --model $model --num_image_embeds 3 \
       --freeze_txt 5 --freeze_img 3   --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed $seed --df true --noise $noise --noise_type Gaussian #Gaussian Salt
    done
  done
done
