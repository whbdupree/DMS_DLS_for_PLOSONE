#!/bin/bash

python simulations.py learn 

python simulations.py reversal &
python simulations.py reversal_prat &
python simulations.py punish &
python simulations.py punish_prat &
python simulations.py devalue  &
wait
