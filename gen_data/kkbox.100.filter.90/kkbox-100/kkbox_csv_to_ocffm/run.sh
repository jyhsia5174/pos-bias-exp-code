#! /bin/bash

./shuf.sh

python3 filter.py
python3 listener_ffm.py
python3 top_song_ffm.py
