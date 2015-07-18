import sys


# -- Stadium Data
# Stadium
# Stadium Capacity
# expanded capacity (standing)
# Location
# Playing surface
# Is Artificial Turf
# Team
# Opened
# Weather
# Station Roof Type
# elevation

import os
from cgm4hadoop import run_cgm_experiment


for line in sys.stdin:
    line = line.strip()
    unpacked = line.split(",")
    #touch(line)
    patient,experiment = line.split(",")
    run_cgm_experiment(patient,experiment)
