#!/bin/bash

python simulations.py reversal02 &
python simulations.py reversal05 &
python simulations.py reversal20 &
python simulations.py punish02 &
python simulations.py punish10 &
python simulations.py punish20 &

wait
