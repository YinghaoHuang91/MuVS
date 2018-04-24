sequence=S1_Box

total=`ls -lah ../Data/HEVA_Validate/$sequence'_1_C1'/Image/*.png | wc -l`
total=$(( total+29 ))
total=$(( total/30 ))

executable=~/VirtualEnvs/tensorflow_13/bin/python 

for ((idx=0; idx<=total; idx++))
do
	echo $idx
	$executable temporal_fit.py $sequence $idx
done
