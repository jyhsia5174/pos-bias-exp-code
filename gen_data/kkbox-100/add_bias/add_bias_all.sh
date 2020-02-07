# Config
de='det.ffm'
pr='prop.ffm'
rd='random.ffm'

python ab_bias.py $rd&
python ab_bias.py $pr&
python ab_bias.py $de&
wait


