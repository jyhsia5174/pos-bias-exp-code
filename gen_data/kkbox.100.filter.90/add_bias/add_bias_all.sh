# Config
D='D.ffm'
DR='DR.ffm'
RD='RD.ffm'

python ab_bias.py $D&
python ab_bias.py $DR&
python ab_bias.py $RD&
wait


