# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:33:25 2019

@author: LowR2
"""

#import os
import saspy
import pandas as pd

def start_session():
    #os.environ["PATH"] += ';'+r'C:\\Program Files\\SASHome\\SASEnterpriseGuide\\7.1'
    sas = saspy.SASsession(cfgname='winiomwin')
    # exports SAS data set to a Pandas data frame
    car_data = sas.sd2df(table='cars', libref='sashelp')
    test_data = sas.sasdata('cars', 'sashelp')
    #print(car_data.head())

def generate_csv():
    print("Generating CSV.................................")
    export_csv = car_data.to_csv(r'C:\Users\LowR2\Desktop\test.csv', index=None, header=True)
    #\\hscdigdcapmdw01\XDrive\SASUsers\LowR2\test.csv   
