#GPU1
sh ./scripts/optimal_toybox/NaiveRehearsal.sh toybox 3
wait
sh ./scripts/optimal_toybox/GEM.sh  toybox 3
wait
sh ./scripts/optimal_toybox/lower_bound.sh  toybox 3
wait
sh ./scripts/optimal_toybox/upper_bound.sh  toybox 3
wait
