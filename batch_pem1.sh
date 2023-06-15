python3 generate_case.py -n 30 -e 0 -p 0 -g 100000 -w 0 -b 0.0 -s 0.1 --name capped_0pv_0ev
pkill helics_broker & helics run --path=runner.json

python3 generate_case.py -n 30 -e 30 -p 0 -g 100000 -w 0 -b 0.0 -s 0.1 --name capped_0pv_30ev
pkill helics_broker & helics run --path=runner.json

python3 generate_case.py -n 30 -e 0 -p 30 -g 100000 -w 0 -b 0.0 -s 0.1 --name capped_30pv_0ev
pkill helics_broker & helics run --path=runner.json
