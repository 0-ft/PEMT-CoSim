python3 generate_case.py -n 30 -e 0 -p 0 -g 99999999 -w 0 -b 0.0 -s 0.1 --name uncapped_0pv_0pv
pkill helics_broker & helics run --path=runner.json

python3 generate_case.py -n 30 -e 30 -p 30 -g 100000 -w 0 -b 0.0 -s 0.1 --name capped_30pv_30ev
pkill helics_broker & helics run --path=runner.json

python3 generate_case.py -n 30 -e 30 -p 30 -g 70000 -w 0 -b 0.0 -s 0.1 --name lowcap_30pv_30ev
pkill helics_broker & helics run --path=runner.json
