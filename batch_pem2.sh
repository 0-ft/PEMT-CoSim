python3 generate_case.py -n 30 -e 0 -p 0 -g 99999999 -w 0 -b 0.0 -s 0.1
pkill helics_broker & helics run --path=runner.json

python3 generate_case.py -n 30 -e 30 -p 30 -g 100000 -w 0 -b 0.0 -s 0.1
pkill helics_broker & helics run --path=runner.json

python3 generate_case.py -n 30 -e 30 -p 30 -g 70000 -w 0 -b 0.0 -s 0.1
pkill helics_broker & helics run --path=runner.json
