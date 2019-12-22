#!/bin/bash

VERSION=$1

if [[ $VERSION == "mod" ]]; then
  echo "======== Modified GraphNAS ========"
  for SEARCH_MODE in macro micro; do
    echo "SEARCH_MODE: ${SEARCH_MODE}"
    for SEED in 123 42 10; do
      for DATASET in Citeseer Cora Pubmed CS Physics Computers Photo; do
        echo "nohup python -u -m graphnas.main --dataset ${DATASET} --optimizer EA --population_size 100 --sample_size 25 --random_seed ${SEED} --search_mode ${SEARCH_MODE} > results_${SEARCH_MODE}_${DATASET}_${SEED}_evolution 2> err_${SEARCH_MODE}_${DATASET}_${SEED}_evolution &"
        # python -u -m graphnas.main \
        #           --dataset ${DATASET} \
        #           --population_size 100 \
        #           --sample_size 25 \
        #           --random_seed ${SEED} \
        #           --search_mode ${SEARCH_MODE} \
        #           > results_${SEARCH_MODE}_${DATASET}_${SEED}_evolution \
        #           2> err_${SEARCH_MODE}_${DATASET}_${SEED}_evolution
      done
    done  
    echo ""
    echo ""
  done  
fi

if [[ $VERSION == "orig" ]]; then
  echo "======== Original GraphNAS ========"
  for SEARCH_MODE in macro micro; do
    echo "SEARCH_MODE: ${SEARCH_MODE}"
    for SEED in 123 42 10; do
      for DATASET in Citeseer Cora Pubmed CS Physics Computers Photo; do
        echo "nohup python -u -m graphnas.main --dataset ${DATASET} --optimizer RL --random_seed ${SEED} --search_mode ${SEARCH_MODE} > results_${SEARCH_MODE}_${DATASET}_${SEED}_RL 2> err_${SEARCH_MODE}_${DATASET}_${SEED}_RL &"
        # python -u -m graphnas.main \
        #           --dataset ${DATASET} \
        #           --random_seed ${SEED} \
        #           --search_mode ${SEARCH_MODE} \
        #           > results_${SEARCH_MODE}_${DATASET}_${SEED}_RL \
        #           2> err_${SEARCH_MODE}_${DATASET}_${SEED}_RL
      done
    done  
    echo ""
    echo ""
  done
fi
