#python3 generate_case.py -e 0 -p 0 -g 2000000 -w 0 -b 0.0 -s -0.1 --name uncapped_0pv_0ev --figure_period 14400
#pkill helics_broker & helics run --path=runner.json


#python3 generate_case.py -e 0 -p 0 -g 80000 -w 0 -b 0.0 -s -0.1 --name capped_0pv_0ev --figure_period 14400
#pkill helics_broker & helics run --path=runner.json
#
#python3 generate_case.py -e 0 -p 30 -g 80000 -w 0 -b 0.0 -s -0.1 --name capped_30pv_0ev --figure_period 14400
#pkill helics_broker & helics run --path=runner.json
#
#python3 generate_case.py -e 30 -p 30 -g 80000 -w 0 -b 0.0 -s -0.1 --name capped_30pv_30ev --figure_period 14400
#pkill helics_broker & helics run --path=runner.json
#
#python3 generate_case.py -e 30 -p 0 -g 80000 -w 0 -b 0.0 -s -0.1 --name capped_0pv_30ev --figure_period 14400
#pkill helics_broker & helics run --path=runner.json


#python3 generate_case.py -e 0 -p 0 -g 100000 -w 0 -b 0.0 -s -0.1 --name highcapped_0pv_0ev --figure_period 14400
#pkill helics_broker & helics run --path=runner.json
#
#python3 generate_case.py -e 0 -p 30 -g 100000 -w 0 -b 0.0 -s -0.1 --name highcapped_30pv_0ev --figure_period 14400
#pkill helics_broker & helics run --path=runner.json
#
#python3 generate_case.py -e 30 -p 30 -g 100000 -w 0 -b 0.0 -s -0.1 --name highcapped_30pv_30ev --figure_period 14400
#pkill helics_broker & helics run --path=runner.json
#
python3 generate_case.py -e 30 -p 0 -g 100000 -w 0 -b 0.0 -s -0.1 --name highcapped_0pv_30ev --figure_period 14400
pkill helics_broker & helics run --path=runner.json
