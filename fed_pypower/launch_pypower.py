import sys

from fncsPYPOWER import pypower_loop

pypower_loop('pypower_config.json','TE_ChallengeH', helicsConfig='pypower_helics_config.json')

