#GPU1
sh ./scripts/optimal_toybox/NaiveRehearsal.sh toybox 3
wait
sh ./scripts/optimal_toybox/GEM.sh  toybox 3
wait
#sh ./scripts/optimal_toybox/NormalNNlower.sh  toybox 3
#wait
#sh ./scripts/optimal_toybox/NormalNNupper.sh  toybox 3
#wait
