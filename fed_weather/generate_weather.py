def generate_weather(tmy_file_name, start_time, stop_time, year):
    import os
    import tesp_support.api as tesp
    # tmy_file_dir = os.getenv('TESP_INSTALL') + '/repository/tesp/data/weather/'
    tmy_file_dir = os.getenv('TESP_INSTALL') + '/share/support/weather/'
    tmy_file = tmy_file_dir + tmy_file_name
    tesp.weathercsv (tmy_file, 'weather.dat', start_time, stop_time, year