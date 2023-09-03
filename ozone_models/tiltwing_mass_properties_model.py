import numpy as np
import csdl
from vehicle_properties.tiltwing_evtol_constants import *



class tiltwing_mass_properties_model(csdl.Model):
    def initialize(self):
        pass

    def define(self):

        mass = get_mass()
        ixx  = get_Ixx()
        iyy  = get_Iyy()
        izz  = get_Izz()
        ixz  = get_Ixz()
        iyz  = get_Iyz()
        cg   = get_cg()

        self.create_input('total_mass', val=mass, shape=(1,))
        self.create_input('ixx', val=ixx, shape=(1,))
        self.create_input('iyy', val=iyy, shape=(1,))
        self.create_input('izz', val=izz, shape=(1,))
        self.create_input('ixz', val=ixz, shape=(1,))
        self.create_input('iyz', val=iyz, shape=(1,))

        self.create_input('cgx', val=cg[0], shape=(1,))
        self.create_input('cgy', val=cg[1], shape=(1,))
        self.create_input('cgz', val=cg[2], shape=(1,))
        