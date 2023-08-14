import numpy as np

'''
This file describes the constants that will be used for the eVTOL concept presented in the following paper:
https://sacd.larc.nasa.gov/wp-content/uploads/sites/167/2022/02/NASA-TM-20210017971.pdf 

The mass properties for this vehicle have been estimated using the paper above as a guideline. 

The mass properties mainly use the following excel sheet to derive the numbers that are used:
https://docs.google.com/spreadsheets/d/1BKzFSKEEyaA976DGK2ruBRD-8pSv9MHeRWSf3x5QqyU/edit#gid=1397306688 

'''

def get_mass():
    tiltwing_mass = 2399.504 # kg
    return tiltwing_mass

def get_cg():
    ''' Defined from the openVSP coordinate frame '''
    cg = np.array([4.0705875499988649,-1.25692571e-16, 2.474650399999944]) # Meters
    return cg

def get_Ixx():
    Ixx = 38411.01146652821
    return Ixx

def get_Iyy():
    Iyy = 83401.42100123441
    return Iyy

def get_Izz():
    Izz = 85331.0429011998
    return Izz

def get_Ixz():
    Ixz = -33003.53641665047
    return Ixz

def get_Ixy():
    Ixy = 0.0
    return Ixy

def get_Iyz():
    Iyz = 0.0
    return Iyz

