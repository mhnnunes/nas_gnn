#!/bin/bash

# This script generates the lines for calling the experiments,
# including all search methods, search spaces, datasets and seeds

VERSION=$1

seeds="10 19 42 79 123"

if [[ $VERSION == "rs" ]]; then
  echo "======== Random Search GraphNAS ========"
  for SEARCH_MODE in macro micro; do
    echo "SEARCH_MODE: ${SEARCH_MODE}"
    for SEED in `echo ${seeds}`; do
      for DATASET in Citeseer Cora Pubmed; do
        echo "nohup python -u -m graphnas.main --dataset ${DATASET} --optimizer RS --random_seed ${SEED} --search_mode ${SEARCH_MODE} > results_${SEARCH_MODE}_${DATASET}_${SEED}_rs 2> err_${SEARCH_MODE}_${DATASET}_${SEED}_rs &"
      done
     for DATASET in Computers CS Photo Physics; do
       echo "nohup python -u -m graphnas.main --dataset ${DATASET} --supervised True --optimizer RS --random_seed ${SEED} --search_mode ${SEARCH_MODE} > results_${SEARCH_MODE}_${DATASET}_${SEED}_evolution 2> err_${SEARCH_MODE}_${DATASET}_${SEED}_evolution &"
     done
    done  
    echo ""
    echo ""
  done  
fi


if [[ $VERSION == "ea" ]]; then
  POP_SIZE=100
  SAMPLE_SIZE=3
  if [[ $2 != "" ]]; then
    POP_SIZE=$2
  fi
  if [[ $3 != "" ]]; then
    SAMPLE_SIZE=$3
  fi
  echo "======== Modified GraphNAS ========"
  for SEARCH_MODE in macro micro; do
    echo "SEARCH_MODE: ${SEARCH_MODE}"
    for SEED in `echo ${seeds}`; do
      for DATASET in Citeseer Cora Pubmed; do
        echo "nohup python -u -m graphnas.main --dataset ${DATASET} --optimizer EA --population_size ${POP_SIZE} --sample_size ${SAMPLE_SIZE} --random_seed ${SEED} --search_mode ${SEARCH_MODE} > results_${SEARCH_MODE}_${DATASET}_${SEED}_ev${POP_SIZE}${SAMPLE_SIZE} 2> err_${SEARCH_MODE}_${DATASET}_${SEED}_ev${POP_SIZE}${SAMPLE_SIZE} &"
      done
     for DATASET in Computers CS Photo Physics; do
       echo "nohup python -u -m graphnas.main --dataset ${DATASET} --supervised True --optimizer EA --population_size ${POP_SIZE} --sample_size ${SAMPLE_SIZE} --random_seed ${SEED} --search_mode ${SEARCH_MODE} > results_${SEARCH_MODE}_${DATASET}_${SEED}_evolution 2> err_${SEARCH_MODE}_${DATASET}_${SEED}_evolution &"
     done
    done  
    echo ""
    echo ""
  done  
fi

if [[ $VERSION == "rl" ]]; then
  echo "======== Original GraphNAS ========"
  for SEARCH_MODE in macro micro; do
    echo "SEARCH_MODE: ${SEARCH_MODE}"
    for SEED in `echo ${seeds}`; do
      for DATASET in Citeseer Cora Pubmed; do
        echo "nohup python -u -m graphnas.main --dataset ${DATASET} --optimizer RL --random_seed ${SEED} --search_mode ${SEARCH_MODE} > results_${SEARCH_MODE}_${DATASET}_${SEED}_RL 2> err_${SEARCH_MODE}_${DATASET}_${SEED}_RL &"
      done
      for DATASET in CS Physics Computers Photo; do
        echo "nohup python -u -m graphnas.main --dataset ${DATASET} --supervised True --optimizer RL --random_seed ${SEED} --search_mode ${SEARCH_MODE} > results_${SEARCH_MODE}_${DATASET}_${SEED}_RL 2> err_${SEARCH_MODE}_${DATASET}_${SEED}_RL &"
      done
    done  
    echo ""
    echo ""
  done
fi
