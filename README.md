# V2G/PET Cosimulation
### How to build and run docker container
1. `docker build -t v2gpet -f docker/Dockerfile .`
2. `docker run -it --mount type=bind,source=<PATH_TO_THIS_DIR>,destination=/PEMT-CoSim --name v2gpet1 v2gpet`

### How to run cosimulation
1. Activate conda environment: `conda activate cosim`
2. Generate EV mobility profiles: `python3 EVProfiles.py` in `fed_ev`
3. Generate simulation scenario: `python3 generate_case.py`
4. Run simulation: `helics run --path=runner.json`
5. Generate figures: `python3 fig.py` in `fed_substation`