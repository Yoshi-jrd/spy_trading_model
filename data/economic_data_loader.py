from fredapi import Fred
import pandas as pd

def load_economic_data():
    fred = Fred(api_key="b4405278e8f6f6f01934fbae7d1953e3")
    gdp_data = fred.get_series('GDP')
    gdp_data.ffill(inplace=True)
    gdp_data.fillna(0, inplace=True)

    cpi_data = fred.get_series('CPIAUCSL')
    cpi_data.ffill(inplace=True)
    cpi_data.fillna(0, inplace=True)

    return gdp_data, cpi_data
