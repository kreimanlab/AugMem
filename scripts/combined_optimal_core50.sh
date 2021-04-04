#!/bin/bash
#GPU0
sh ./scripts/optimal_core50/EWC.sh core50 0
wait
sh ./scripts/optimal_core50/L2.sh core50 0
wait
sh ./scripts/optimal_core50/MAS.sh core50 0
wait
sh ./scripts/optimal_core50/SI.sh core50 0
wait
sh ./scripts/optimal_core50/iCARL.sh core50 0
wait
sh ./scripts/optimal_core50/NaiveRehearsal.sh core50 0
wait
sh ./scripts/optimal_core50/GEM.sh  core50 0
wait
sh ./scripts/optimal_core50/lower_bound.sh  core50 0
wait
sh ./scripts/optimal_core50/upper_bound.sh  core50 0
wait
sh ./scripts/optimal_core50/AugMem.sh  core50 0
wait
