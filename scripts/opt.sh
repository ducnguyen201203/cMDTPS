#find best param for mlm

#!/bin/bash
echo -e "\n\n\t\t==========Finetune new mlm===========" 
for i in $(seq 10 5 100)
do
    echo -e "\t\t\t========change mlm prob to $i =============" 
    python run.py --cfg /home/k64t/person-search/DAProject/UET/config_model.yml --l-names id mlm sdm --l-mlm-use-custom --l-mlm-prob $i
done


#find best param for mim
echo -e "\n\n\t\t==========Finetune new MIM==========" 

#find best param for triolet
echo -e "\n\n\t\t==========Finetune new Triplet===========" 


#find best ưeight for lossess
##lamda 1 = loss mlm

##lamda 2 = loss mim

##lamda 3 = loss trip
