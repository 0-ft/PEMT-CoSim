python3 generate_case.py -n 30 -e 0 -p 0 -g 2000000
pkill helics_broker & helics run --path=runner.json

python3 generate_case.py -n 30 -e 0 -p 0 -g 80000
pkill helics_broker & helics run --path=runner.json

python3 generate_case.py -n 30 -e 30 -p 0 -g 80000
pkill helics_broker & helics run --path=runner.json

python3 generate_case.py -n 30 -e 0 -p 30 -g 80000
pkill helics_broker & helics run --path=runner.json

python3 generate_case.py -n 30 -e 30 -p 30 -g 80000
pkill helics_broker & helics run --path=runner.json
