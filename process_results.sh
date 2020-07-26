#!/bin/bash

DATASET=$1
DIR=$2
TYPE=$3

if [[ $TYPE == ""  ]]; then
  cat ${DIR}/results_evolution_${DATASET} | grep parent | cut -d '|' -f1 | cut -d':' -f2,3 | sed -e "s/^\ *//g" | sed -e "s/\ *val_score:\ */>/g"  > ${DIR}/parent_temp
  cat ${DIR}/results_evolution_${DATASET} | grep parent | cut -d '|' -f2 | cut -d':' -f2,3 | sed -e "s/^\ *//g" | sed -e "s/\ ,\ *val_score:\ */>/g" > ${DIR}/child_temp
  paste ${DIR}/parent_temp ${DIR}/child_temp -d'>' > ${DIR}/parent_child_${DATASET}
  rm ${DIR}/parent_temp ${DIR}/child_temp
  cat ${DIR}/results_evolution_${DATASET} | grep -E '(time|Time)' > ${DIR}/time_${DATASET}
  START_INITIAL=`grep -n initial\ random\ population\ =  ${DIR}/results_evolution_${DATASET} | cut -d':' -f 1`
  END_INITIAL=`grep -n initial\ random\ population\ D  ${DIR}/results_evolution_${DATASET} | cut -d':' -f 1`
  DIFF=`echo  $END_INITIAL - $START_INITIAL  + 1 | bc`
  head -n ${END_INITIAL} ${DIR}/results_evolution_${DATASET} | tail -n ${DIFF} | grep val_score | cut -d':' -f3 > ${DIR}/initial_population_${DATASET}
  echo "Mean,Median,Best" > ${DIR}/population_stats_${DATASET}
  cat ${DIR}/results_evolution_${DATASET} | grep STATS | awk '{print $4,$5,$6}' | sed -e 's/\ /,/g' >> ${DIR}/population_stats_${DATASET}
elif [[ $TYPE == "RL" ]]; then
  if [[ `echo ${DIR} | grep micro | wc -l` -gt 0 ]]; then
    ACTIONS=`cat ${DIR}/results_${TYPE}_${DATASET}  | grep "best results" | cut -d':' -f 3,4 | sed -e "s/\],\ 'hyper_param':\ \[/,\ /g" | sed -e "s/^\ //g" | sed -e "s/\}//g"`
    ACC=`cat ${DIR}/results_${TYPE}_${DATASET}  | grep "best results" | cut -d':' -f5 | sed -e "s/^\ //g" | sed -e "s/\ +\/-\ />/g"`
  else
    ACTIONS=`cat ${DIR}/results_${TYPE}_${DATASET}  | grep "best results" | cut -d':' -f2`
    ACC=`cat ${DIR}/results_${TYPE}_${DATASET}  | grep "best results" | cut -d':' -f3 | sed -e "s/^\ //g" | sed -e "s/\ +\/-\ />/g"`
  fi
  echo "${ACTIONS}>${ACC}" > ${DIR}/best_acc_${TYPE}_${DATASET}
elif [[ $TYPE == "RS" ]]; then
  cat ${DIR}/results_${TYPE}_${DATASET} | grep ^val_score | cut -d',' -f1 | cut -d':' -f2 > ${DIR}/val_scores_${TYPE}_${DATASET}
  cat ${DIR}/results_${TYPE}_${DATASET} | grep time | cut -d':' -f2 | sed -e 's/\ //g' > ${DIR}/time_${TYPE}_${DATASET}
  ACC=`cat ${DIR}/results_${TYPE}_${DATASET} | grep BEST | grep Accuracy | cut -d':' -f2 | sed -e "s/^\ *//g"`
  ACTIONS=`cat ${DIR}/results_${TYPE}_${DATASET} | grep BEST | grep Actions | cut -d':' -f2 | sed -e "s/^\ *//g" | sed -e "s/\[\[/\[/g" | sed -e "s/\]\]/\]/g"`
  echo "${ACTIONS}>${ACC}" > ${DIR}/best_acc_${TYPE}_${DATASET}
else
  cat ${DIR}/results_${TYPE}_${DATASET} | grep parent | cut -d '|' -f1 | cut -d':' -f2,3 | sed -e "s/^\ *//g" | sed -e "s/\ *val_score:\ */>/g"  > ${DIR}/parent_temp
  cat ${DIR}/results_${TYPE}_${DATASET} | grep parent | cut -d '|' -f2 | cut -d':' -f2,3 | sed -e "s/^\ *//g" | sed -e "s/\ ,\ *val_score:\ */>/g" > ${DIR}/child_temp
  paste ${DIR}/parent_temp ${DIR}/child_temp -d'>' > ${DIR}/parent_child_${DATASET}_${TYPE} && sed -i "s/\ >/>/g"  ${DIR}/parent_child_${DATASET}_${TYPE}
  rm ${DIR}/parent_temp ${DIR}/child_temp
  cat ${DIR}/results_${TYPE}_${DATASET} | grep -E '(time|Time)' > ${DIR}/time_${DATASET}_${TYPE}
  START_INITIAL=`grep -n initial\ random\ population\ =  ${DIR}/results_${TYPE}_${DATASET} | cut -d':' -f 1`
  END_INITIAL=`grep -n initial\ random\ population\ D  ${DIR}/results_${TYPE}_${DATASET} | cut -d':' -f 1`
  DIFF=`echo  $END_INITIAL - $START_INITIAL  + 1 | bc`
  head -n ${END_INITIAL} ${DIR}/results_${TYPE}_${DATASET} | tail -n ${DIFF} | grep val_score | cut -d':' -f3 > ${DIR}/initial_population_${DATASET}_${TYPE}
  echo "Mean,Median,Best" > ${DIR}/population_stats_${DATASET}_${TYPE}
  cat ${DIR}/results_${TYPE}_${DATASET} | grep STATS | awk '{print $4,$5,$6}' | sed -e 's/\ /,/g' >> ${DIR}/population_stats_${DATASET}_${TYPE}
fi
