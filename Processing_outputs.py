#This function compresses and processes the outputs from MC_calc.py ready for plotting by Plot_processed_outputs.py.

import numpy as np
import pylab
from scipy.integrate import *
from scipy.interpolate import interp1d
from stellar_funs import main_sun_fun
from scipy import optimize
import scipy.optimize 
from all_classes_Ledits import *
import time
from other_functions import *

new_inputs = []
inputs_for_MC=[]
Total_Fe_array = []

global g,rp,mp
def use_one_output(inputs,MCinputs):
    k= 0
    global g,rp,mp
    while k<len(inputs):       
        if (np.size(inputs[k])==1) and (inputs[k].total_time[-1] > 7.4e9):
            #print(inputs[k].total_time[-1])
            top = (inputs[k].total_y[13][0] + inputs[k].total_y[12][0]) 
            bottom =  (inputs[k].total_y[1][0] + inputs[k].total_y[0][0] )
            CO2_H2O_ratio = top/bottom
            
            rp = 6371000*MCinputs[k][1].RE
            mp = 5.972e24*MCinputs[k][1].ME
            g = 6.67e-11*mp/(rp**2)
            Pbase = (inputs[k].Pressre_H2O[-1] + inputs[k].CO2_Pressure_array[-1] + inputs[k].total_y[22][-1])
            Mvolatiles = (Pbase*4*np.pi*rp**2)/g 
            new_inputs.append(inputs[k])
            inputs_for_MC.append(MCinputs[k])
            Total_Fe_array.append(MCinputs[k][1].Total_Fe_mol_fraction)

        k= k+1 

### Load outputs and inputs. Note it is possible to load multiple output files and process them all at once

inputs = np.load('MC_outputs_Trappist.npy',allow_pickle = True) # 
MCinputs = np.load('MC_inputs_Trappist.npy',allow_pickle = True) 
use_one_output(inputs,MCinputs) 

#inputs = np.load('MC_outputs_Trappist_ii.npy',allow_pickle = True) # 
#MCinputs = np.load('MC_inputs_Trappist_ii.npy',allow_pickle = True) 
#use_one_output(inputs,MCinputs) 

# different N2 pressures
ppN2 = 1e5


print(np.shape(new_inputs))
inputs = np.array(new_inputs)

def interpolate_class(saved_outputs):
    outs=[]
    for i in range(0,len(saved_outputs)):
        time_starts = np.min([np.where(saved_outputs[i].total_time>1)])-1
        time = saved_outputs[i].total_time[time_starts:]
        num_t_elements = 1000 
        new_time = np.logspace(np.log10(np.min(time[:])),np.max([np.log10(time[:-1])]),num_t_elements)
        num_y = 32
        new_total_y = np.zeros(shape=(num_y,len(new_time)))
        for k in range(0,num_y):
            f1 = interp1d(time,saved_outputs[i].total_y[k][time_starts:])
            new_total_y[k] = f1(new_time)
 
        f1 = interp1d(time,saved_outputs[i].FH2O_array[time_starts:])
        new_FH2O_array = f1(new_time)

        f1 = interp1d(time,saved_outputs[i].FCO2_array[time_starts:])
        new_FCO2_array = f1(new_time)
        
        f1 = interp1d(time,saved_outputs[i].MH2O_liq[time_starts:])
        new_MH2O_liq = f1(new_time)    

        f1 = interp1d(time,saved_outputs[i].MH2O_crystal[time_starts:])
        new_MH2O_crystal = f1(new_time)    
  
        f1 = interp1d(time,saved_outputs[i].MCO2_liq[time_starts:])
        new_MCO2_liq = f1(new_time)    

        f1 = interp1d(time,saved_outputs[i].Pressre_H2O[time_starts:])
        new_Pressre_H2O = f1(new_time)    

        f1 = interp1d(time,saved_outputs[i].CO2_Pressure_array[time_starts:])
        new_CO2_Pressure_array = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].fO2_array[time_starts:])
        new_fO2_array = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].Mass_O_atm[time_starts:])
        new_Mass_O_atm = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].Mass_O_atm[time_starts:])
        new_Mass_O_atm = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].Mass_O_dissolved[time_starts:])
        new_Mass_O_dissolved = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].water_frac[time_starts:])
        new_water_frac = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].Ocean_depth[time_starts:])
        new_Ocean_depth = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].Max_depth[time_starts:])
        new_Max_depth = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].Ocean_fraction[time_starts:])
        new_Ocean_fraction = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].TOA_shortwave[time_starts:]) 
        new_TOA_shortwave = f1(new_time)  

        output_class = Model_outputs(new_time,new_total_y,new_FH2O_array,new_FCO2_array,new_MH2O_liq,new_MH2O_crystal,new_MCO2_liq,new_Pressre_H2O,new_CO2_Pressure_array,new_fO2_array,new_Mass_O_atm,new_Mass_O_dissolved,new_water_frac,new_Ocean_depth,new_Max_depth,new_Ocean_fraction,new_TOA_shortwave) 
        outs.append(output_class)
    return outs

inputs = interpolate_class(inputs)


################################################################################
## Post-processing of outputs:
# Oxygen fugacity and mantle redox functions (for post-processing)
def buffer_fO2(T,Press,redox_buffer): # T in K, P in bar
    if redox_buffer == 'FMQ':
        [A,B,C] = [25738.0, 9.0, 0.092]
    elif redox_buffer == 'IW':
        [A,B,C] = [27215 ,6.57 ,0.0552]
    elif redox_buffer == 'MH':
        [A,B,C] = [25700.6,14.558,0.019] # from Frost
    else:
        print ('error, no such redox buffer')
        return -999
    return 10**(-A/T + B + C*(Press-1)/T)

def get_fO2(XFe2O3_over_XFeO,P,T,Total_Fe): ## Total_Fe is a mole fraction of iron minerals XFeO + XFeO1.5 = Total_Fe, and XFe2O3 = 0.5*XFeO1.5, xo XFeO + 2XFe2O3 = Total_Fe
    XAl2O3 = 0.022423 
    XCaO = 0.0335 
    XNa2O = 0.0024 
    XK2O = 0.0001077 
    terms1 =  11492.0/T - 6.675 - 2.243*XAl2O3
    terms2 = 3.201*XCaO + 5.854 * XNa2O
    terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
    terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
    fO2 =  np.exp( (np.log(XFe2O3_over_XFeO) + 1.828 * Total_Fe -(terms1+terms2+terms3+terms4) )/0.196)
    return fO2  

total_time = []
total_y = []
FH2O_array=  []
FCO2_array= []
MH2O_liq =  []
MCO2_liq =  []
Pressre_H2O = []
CO2_Pressure_array =  []
fO2_array =  []
Mass_O_atm =  []
Mass_O_dissolved =  []
water_frac =  []
Ocean_depth =  []
Max_depth = []
Ocean_fraction =  []
MH2O_crystal   = []
TOA_L = [] 

for k in range(0,len(inputs)):
    total_time.append( inputs[k].total_time )
    total_y.append(inputs[k].total_y)
    FH2O_array.append(inputs[k].FH2O_array )
    FCO2_array.append(inputs[k].FCO2_array )
    MH2O_liq.append(inputs[k].MH2O_liq )
    MCO2_liq.append(inputs[k].MCO2_liq )
    Pressre_H2O.append(inputs[k].Pressre_H2O) 
    MH2O_crystal.append(inputs[k].MH2O_crystal) 
    CO2_Pressure_array.append(inputs[k].CO2_Pressure_array )
    fO2_array.append(inputs[k].fO2_array )
    Mass_O_atm.append(inputs[k].Mass_O_atm )
    Mass_O_dissolved.append(inputs[k].Mass_O_dissolved )
    water_frac.append(inputs[k].water_frac )
    Ocean_depth.append(inputs[k].Ocean_depth )
    Max_depth.append(inputs[k].Max_depth )
    Ocean_fraction.append(inputs[k].Ocean_fraction )
    TOA_L.append(inputs[k].TOA_shortwave) 


f_O2_FMQ = []
f_O2_IW = []
f_O2_MH = []
f_O2_mantle = []
iron_ratio = []
iron_ratio_norm = []
actual_phi_surf_melt_ar = []
XH2O_melt = []
XCO2_melt = []

rc = MCinputs[0][1].rc
mantle_mass = 0.0

x_low = 1.0 #Start time for plotting (years)
x_high =np.max(total_time[0])+0.5e9 #Finish time for plotting (years)

mantleCO2_totalCO2 = []
mantleH2O_totalH2O = []

MO_Oxy_sink = []

rp = 6371000*MCinputs[0][1].RE
ll = rp - rc
alpha = 2e-5
kappa = 1e-6
Racr = 1.1e3

F_H2O_new = []
F_CO2_new = []
F_CO_new = []
F_H2_new = []
F_CH4_new = []
F_SO2_new = []
F_H2S_new = []
F_S2_new = []
O2_consumption_new = []
True_WW = []
True_WW_finalO2 = []

mantle_CO2_fraction = []
mantle_H2O_fraction=[]

Melt_volume = np.copy(total_time)
Plate_velocity = np.copy(total_time)

integrate_XUV = []

for k in range(0,len(inputs)):

    #stellar XUV
    Start_time = inputs_for_MC[k][0].Start_time
    Max_time=np.max([inputs_for_MC[k][3].tfin0,inputs_for_MC[k][3].tfin1,inputs_for_MC[k][3].tfin2,inputs_for_MC[k][3].tfin3,inputs_for_MC[k][3].tfin4]) #Model end time
    new_t = np.linspace(Start_time/1e9,Max_time/1e9,100000)
    Stellar_Mass = inputs_for_MC[k][4].Stellar_Mass
    tsat_XUV = inputs_for_MC[k][4].tsat_XUV #XUV saturation time
    fsat = inputs_for_MC[k][4].fsat 
    beta0 = inputs_for_MC[k][4].beta0
    Planet_sep = inputs_for_MC[k][1].Planet_sep #planet-star separation (AU)
    [Relative_total_Lum,Relative_XUV_lum,Absolute_total_Lum,Absolute_XUV_Lum] = main_sun_fun(new_t,Stellar_Mass,tsat_XUV,beta0,fsat) #Calculate stellar evolution
    AbsXUV = Absolute_XUV_Lum/(4*np.pi*(Planet_sep*1.496e11)**2) #XUV function, used to calculate XUV-driven escape
    the_x_axis = new_t*1e9*365*24*60*60
    integrate_XUV.append(scipy.integrate.trapz(AbsXUV,the_x_axis))

    f_O2_FMQ.append(inputs[k].Ocean_fraction*0 )
    f_O2_IW.append(inputs[k].Ocean_fraction*0 )
    f_O2_MH.append(inputs[k].Ocean_fraction*0 )
    f_O2_mantle.append(inputs[k].Ocean_fraction*0 )
    iron_ratio.append(inputs[k].Ocean_fraction*0 )
    iron_ratio_norm.append(inputs[k].Ocean_fraction*0 )
    actual_phi_surf_melt_ar.append(inputs[k].Ocean_fraction*0 )
    XH2O_melt.append(inputs[k].Ocean_fraction*0 )
    XCO2_melt.append(inputs[k].Ocean_fraction*0 )
    F_H2O_new.append(inputs[k].Ocean_fraction*0 )
    F_CO2_new.append(inputs[k].Ocean_fraction*0 )
    F_CO_new.append(inputs[k].Ocean_fraction*0 )
    F_H2_new.append(inputs[k].Ocean_fraction*0 )
    F_CH4_new.append(inputs[k].Ocean_fraction*0 )
    F_SO2_new.append(inputs[k].Ocean_fraction*0 )
    F_H2S_new.append(inputs[k].Ocean_fraction*0 )
    F_S2_new.append(inputs[k].Ocean_fraction*0 )
    O2_consumption_new.append(inputs[k].Ocean_fraction*0 )
    
    MO_Oxy_sink.append(inputs[k].Ocean_fraction*0 )

    mantle_CO2_fraction.append(inputs[k].Ocean_fraction*0 ) 
    mantle_H2O_fraction.append(inputs[k].Ocean_fraction*0 )

    terminalMO_index = 0
    true_WW_index = 0
    for i in range(0,len(total_time[k])):

        mantle_CO2_fraction[k][i] = total_y[k][13][i]/(total_y[k][13][i]+total_y[k][12][i])
        mantle_H2O_fraction[k][i] = total_y[k][0][i]/(total_y[k][0][i]+total_y[k][1][i])

        Pressure_surface =fO2_array[k][i] + Pressre_H2O[k][i]*water_frac[k][i] + CO2_Pressure_array[k][i] + ppN2 

        f_O2_FMQ[k][i] = buffer_fO2(total_y[k][7][i],Pressure_surface/1e5,'FMQ')
        f_O2_IW[k][i] = buffer_fO2(total_y[k][7][i],Pressure_surface/1e5,'IW')
        f_O2_MH[k][i] =  buffer_fO2(total_y[k][7][i],Pressure_surface/1e5,'MH')
        iron3 = total_y[k][5][i]*56/(56.0+1.5*16.0)
        iron2 = total_y[k][6][i]*56/(56.0+16.0)
        iron_ratio[k][i] = iron3/iron2
        iron_ratio_norm[k][i] = iron3/(iron2+iron3)
        f_O2_mantle[k][i] = get_fO2(0.5*iron3/iron2,Pressure_surface,total_y[k][7][i],Total_Fe_array[k])
        T_for_melting = float(total_y[k][7][i])
        Poverburd = fO2_array[k][i] + Pressre_H2O[k][i] + CO2_Pressure_array[k][i] + ppN2  
        alpha = 2e-5
        cp = 1.2e3 
        pm = 4000.0
        rdck = optimize.minimize(find_r,x0=float(total_y[k][2][i]),args = (T_for_melting,alpha,g,cp,pm,rp,float(Poverburd),0))
        rad_check = float(rdck.x[0])
        if rad_check>rp:
            rad_check = rp
        [actual_phi_surf_melt,actual_visc,Va] = temp_meltfrac(0.99998*rad_check,rp,alpha,pm,T_for_melting,cp,g,Poverburd,0)
        actual_phi_surf_melt_ar[k][i]= actual_phi_surf_melt
        F = actual_phi_surf_melt_ar[k][i]
        x = 0.01550152865954013
        M_H2O = 18.01528
        M_CO2 = 44.01
        mantle_mass = (4./3. * np.pi * pm * (rp**3 - rc**3))
        XH2O_melt_max = x*M_H2O*0.499 
        XCO2_melt_max = x*M_CO2*0.499 
        if F >0:
            XH2O_melt[k][i] = np.min([0.99*XH2O_melt_max,(1- (1-F)**(1/0.01)) * (total_y[k][0][i]/mantle_mass)/F ]) 
            XCO2_melt[k][i] =  np.min([0.99*XCO2_melt_max,(1- (1-F)**(1/2e-3)) * (total_y[k][13][i]/mantle_mass)/F ])
        else:
            XH2O_melt[k][i] = 0.0 
            XCO2_melt[k][i] =  0.0

        if (total_y[k][4][i] <= np.max(total_y[k][4][i:]))and(total_y[k][22][i] <= 1.0 )and(true_WW_index<1):
            True_WW.append(inputs_for_MC[k][5].Tstrat)
            True_WW_finalO2.append(total_y[k][22][-1]/1e5)
            true_WW_index = 50.0

        MO_Oxy_sink[k] = -np.gradient(Mass_O_dissolved[k],total_time[k]) 


    Q =  total_y[k][11]
    Aoc = 4*np.pi*rp**2
    for i in range(0,len(Q)):
        if total_y[k][16][i]/1000 < 1e-11:
            total_y[k][16][i] = 0.0
                
    Melt_volume[k] = total_y[k][25] #(Q*4*np.pi*rp**2)**2/(2*4.2*(total_y[k][7] -total_y[k][8] ))**2 * (np.pi*kappa)/(Aoc)  *total_y[k][16] 
    Plate_velocity[k] = 365*24*60*60*Melt_volume[k]/(total_y[k][16]  * 3 * np.pi*rp)
    for i in range(0,len(Q)):
        if total_y[k][16][i]/1000 < 1e-11:
            Plate_velocity[k][i] = 0    


        Pressure_surface =fO2_array[k][i] + Pressre_H2O[k][i]*water_frac[k][i] + CO2_Pressure_array[k][i] + ppN2  
        melt_mass = Melt_volume[k][i]*4000*1000
        [F_H2O,F_CO2,F_H2,F_CO,F_CH4,F_SO2,F_H2S,F_S2,O2_consumption] = [0,0,0,0,0,0,0,0,0]
        F_H2O_new[k][i] = F_H2O*365*24*60*60/(1e12)
        F_CO2_new[k][i] = F_CO2*365*24*60*60/(1e12)
        F_CO_new[k][i] = F_CO*365*24*60*60/(1e12)
        F_H2_new[k][i] = F_H2*365*24*60*60/(1e12)
        F_CH4_new[k][i] = F_CH4*365*24*60*60/(1e12)
        F_SO2_new[k][i] = F_SO2*365*24*60*60/(1e12)
        F_H2S_new[k][i] = F_H2S*365*24*60*60/(1e12)
        F_S2_new[k][i] = F_S2*365*24*60*60/(1e12)
        O2_consumption_new[k][i] = O2_consumption*365*24*60*60/(1e12)
## End post-processing of outputs:
################################################################################

np.savez('Compressed_TrappistOutputs', inputs_for_MC=inputs_for_MC, Total_Fe_array=Total_Fe_array, total_time=total_time, total_y=total_y,FH2O_array =FH2O_array, FCO2_array =FCO2_array,MH2O_liq=MH2O_liq, MCO2_liq=MCO2_liq , Pressre_H2O=Pressre_H2O,MH2O_crystal=MH2O_crystal,CO2_Pressure_array=CO2_Pressure_array,fO2_array=fO2_array,Mass_O_atm=Mass_O_atm,Mass_O_dissolved=Mass_O_dissolved,water_frac=water_frac,Ocean_depth=Ocean_depth,Max_depth=Max_depth,Ocean_fraction=Ocean_fraction,TOA_L=TOA_L, Plate_velocity=Plate_velocity,Melt_volume=Melt_volume,O2_consumption_new=O2_consumption_new, actual_phi_surf_melt_ar=actual_phi_surf_melt_ar, f_O2_mantle=f_O2_mantle , iron_ratio_norm=iron_ratio_norm, iron_ratio=iron_ratio, f_O2_MH=f_O2_MH, f_O2_IW=f_O2_IW, f_O2_FMQ=f_O2_FMQ, mantle_H2O_fraction=mantle_H2O_fraction, mantle_CO2_fraction=mantle_CO2_fraction, integrate_XUV=integrate_XUV,rp=rp)



