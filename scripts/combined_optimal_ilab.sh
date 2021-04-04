#!/bin/bash
#GPU0
sh ./scripts/optimal_ilab/EWC.sh ilab2mlight 2
wait
sh ./scripts/optimal_ilab/L2.sh ilab2mlight 2
wait
sh ./scripts/optimal_ilab/MAS.sh ilab2mlight 2
wait
sh ./scripts/optimal_ilab/SI.sh ilab2mlight 2
wait
sh ./scripts/optimal_ilab/iCARL.sh ilab2mlight 2
wait
