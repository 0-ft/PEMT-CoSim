import json
import resource
import sys

sys.path.append('..')

from my_tesp_support_api.utils import DotDict


class FederateHelper:
    def __init__(self, configfile):

        with open(configfile, encoding='utf-8') as f:
            self.agents = DotDict(json.loads(f.read()))  # federate_config is the dict data structure
            f.close()

        self.house_name_list = list(self.agents.houses.keys())
        self.billingmeter_name_list = [self.agents.houses[house]['billingmeter_id'] for house in self.house_name_list]
        self.hvac_name_list = [houseName + '_hvac' for houseName in self.house_name_list]
        self.inverter_name_list = list(self.agents.inverters.keys())

        self.housesInfo_dict = {}  # where the house name is the key,
        # each item includes 5 key:value for
        # 'meter': name of the triplex meter for the house
        # 'VPP'  : name of the VPP
        # 'HVAC' : name of the HVAC
        # 'PV'   : name of the PV inverter; if no PV, None
        # 'battery': name of battery inverter; if no battery, None
        for i, house_name in enumerate(self.agents.houses.keys()):
            self.housesInfo_dict[house_name] = {}
            hvac_name = self.hvac_name_list[i]
            meter = self.billingmeter_name_list[i]
            vpp = self.agents.houses[house_name]['house_class']
            self.housesInfo_dict[house_name]['VPP'] = vpp
            self.housesInfo_dict[house_name]['meter'] = meter
            self.housesInfo_dict[house_name]['HVAC'] = hvac_name
            self.housesInfo_dict[house_name]['PV'] = None
            self.housesInfo_dict[house_name]['battery'] = None

        for key, dict in self.agents.inverters.items():
            billingmeter_id = dict['billingmeter_id']
            if billingmeter_id in self.billingmeter_name_list:
                house_name = self.house_name_list[self.billingmeter_name_list.index(billingmeter_id)]
                resource = dict['resource']
                if resource == 'solar':
                    self.housesInfo_dict[house_name]['PV'] = key
                if resource == 'battery':
                    self.housesInfo_dict[house_name]['battery'] = key

        # initialize objects for subscriptions and the publications, and save these objects in dictionary
        # for House

    def show_resource_consumption(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        RESOURCES = [
            ('ru_utime', 'User time'),
            ('ru_stime', 'System time'),
            ('ru_maxrss', 'Max. Resident Set Size'),
            ('ru_ixrss', 'Shared Memory Size'),
            ('ru_idrss', 'Unshared Memory Size'),
            ('ru_isrss', 'Stack Size'),
            ('ru_inblock', 'Block inputs'),
            ('ru_oublock', 'Block outputs')]
        print('Resource usage:')
        for name, desc in RESOURCES:
            print('  {:<25} ({:<10}) = {}'.format(desc, name, getattr(usage, name)))