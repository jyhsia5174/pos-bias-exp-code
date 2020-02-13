awk '{print $1} ' det.ffm | grep ':1:' |  wc -l
awk '{print $1} ' prop.ffm | grep ':1:' |  wc -l
awk '{print $1} ' random.ffm | grep ':1:' |  wc -l
awk '{print $1} ' det.ffm.pos.bias | grep ':1:' |  wc -l
awk '{print $1} ' prop.ffm.pos.bias | grep ':1:' |  wc -l
awk '{print $1} ' random.ffm.pos.bias | grep ':1:' |  wc -l

