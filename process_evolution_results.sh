#!/bin/bash

DATABASE=$1
DIR=$2

cat ${DIR}/results_evolution_${DATABASE} | grep parent | cut -d '|' -f1 | cut -d':' -f3 | sed 's/\ //g'  > ${DIR}/parent_temp

cat ${DIR}/results_evolution_${DATABASE} | grep parent | cut -d '|' -f2 | cut -d':' -f3 > ${DIR}/child_temp

paste ${DIR}/parent_temp ${DIR}/child_temp -d',' > ${DIR}/parent_child_${DATABASE}

rm ${DIR}/parent_temp ${DIR}/child_temp

cat ${DIR}/results_evolution_${DATABASE} | grep -E '(time|Time)' > ${DIR}/time_${DATABASE}

START_INITIAL=`grep -n initial\ random\ population\ =  ${DIR}/results_evolution_${DATABASE} | cut -d':' -f 1`
END_INITIAL=`grep -n initial\ random\ population\ D  ${DIR}/results_evolution_${DATABASE} | cut -d':' -f 1`
DIFF=`echo  $END_INITIAL - $START_INITIAL  + 1 | bc`
head -n ${END_INITIAL} ${DIR}/results_evolution_${DATABASE} | tail -n ${DIFF} | grep val_score | cut -d':' -f3 > ${DIR}/initial_population_${DATABASE}

echo "Mean,Median,Best" > ${DIR}/population_stats_${DATABASE}
cat ${DIR}/results_evolution_${DATABASE} | grep STATS | awk '{print $4,$5,$6}' | sed -e 's/\ /,/g' >> ${DIR}/population_stats_${DATABASE}

