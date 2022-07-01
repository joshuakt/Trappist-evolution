## This code takes the compressed outputs from "Processing_outputs.py" and plots them. Options are available to restrict the plotted outputs to those that satisfy current mass-radius constraints (which constrain surface volatiles), following Agol et al. Options are also available for reproducing other sensitivity tests e.g. Venus-like oxygen sinks, Earth-like initial volatile inventories.

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
import pdb

Processed_everything = np.load("Compressed_TrappistOutputs.npz",allow_pickle=True)

inputs_for_MC=Processed_everything['inputs_for_MC']
Total_Fe_array=Processed_everything['Total_Fe_array']
total_time=Processed_everything['total_time']
total_y=Processed_everything['total_y']
FH2O_array =Processed_everything['FH2O_array']
FCO2_array =Processed_everything['FCO2_array']
MH2O_liq=Processed_everything['MH2O_liq']
MCO2_liq=Processed_everything['MCO2_liq']
Pressre_H2O=Processed_everything['Pressre_H2O']
MH2O_crystal=Processed_everything['MH2O_crystal']
CO2_Pressure_array=Processed_everything['CO2_Pressure_array']
fO2_array=Processed_everything['fO2_array']
Mass_O_atm=Processed_everything['Mass_O_atm']
Mass_O_dissolved=Processed_everything['Mass_O_dissolved']
water_frac=Processed_everything['water_frac']
Ocean_depth=Processed_everything['Ocean_depth']
Max_depth=Processed_everything['Max_depth']
Ocean_fraction=Processed_everything['Ocean_fraction']
TOA_L=Processed_everything['TOA_L']
Plate_velocity=Processed_everything['Plate_velocity']
Melt_volume=Processed_everything['Melt_volume']
O2_consumption_new=Processed_everything['O2_consumption_new']
actual_phi_surf_melt_ar=Processed_everything['actual_phi_surf_melt_ar']
f_O2_mantle=Processed_everything['f_O2_mantle']
iron_ratio_norm=Processed_everything['iron_ratio_norm']
iron_ratio=Processed_everything['iron_ratio']
f_O2_MH=Processed_everything['f_O2_MH']
f_O2_IW=Processed_everything['f_O2_IW']
f_O2_FMQ=Processed_everything['f_O2_FMQ']
mantle_H2O_fraction=Processed_everything['mantle_H2O_fraction']
mantle_CO2_fraction=Processed_everything['mantle_CO2_fraction']
integrate_XUV=Processed_everything['integrate_XUV']
rp=Processed_everything['rp']

optional_second="n"

if optional_second=="y": #option for combining multiple processed outputs
    Processed_everything2 = np.load("Compressed_TrappistOutputs_ii.npz",allow_pickle=True)

    inputs_for_MC2=Processed_everything2['inputs_for_MC']
    inputs_for_MC = np.concatenate([inputs_for_MC,inputs_for_MC2])
    Total_Fe_array2=Processed_everything2['Total_Fe_array']
    Total_Fe_array = np.concatenate([Total_Fe_array,Total_Fe_array2])
    total_time2=Processed_everything2['total_time']
    total_time = np.concatenate([total_time,total_time2])
    total_y2=Processed_everything2['total_y']
    total_y = np.concatenate([total_y,total_y2])
    FH2O_array2=Processed_everything2['FH2O_array']
    FH2O_array = np.concatenate([FH2O_array,FH2O_array2])
    FCO2_array2=Processed_everything2['FCO2_array']
    FCO2_array = np.concatenate([FCO2_array,FCO2_array2])
    MH2O_liq2=Processed_everything2['MH2O_liq']
    MH2O_liq = np.concatenate([MH2O_liq,MH2O_liq2])
    MCO2_liq2=Processed_everything2['MCO2_liq']
    MCO2_liq = np.concatenate([MCO2_liq,MCO2_liq2])
    Pressre_H2O2=Processed_everything2['Pressre_H2O']
    Pressre_H2O = np.concatenate([Pressre_H2O,Pressre_H2O2])
    MH2O_crystal2=Processed_everything2['MH2O_crystal']
    MH2O_crystal = np.concatenate([MH2O_crystal,MH2O_crystal2])
    CO2_Pressure_array2=Processed_everything2['CO2_Pressure_array']
    CO2_Pressure_array = np.concatenate([CO2_Pressure_array,CO2_Pressure_array2])
    fO2_array2=Processed_everything2['fO2_array']
    fO2_array = np.concatenate([fO2_array,fO2_array2])
    Mass_O_atm2=Processed_everything2['Mass_O_atm']
    Mass_O_atm = np.concatenate([Mass_O_atm,Mass_O_atm2])
    Mass_O_dissolved2=Processed_everything2['Mass_O_dissolved']
    Mass_O_dissolved = np.concatenate([Mass_O_dissolved,Mass_O_dissolved2])
    water_frac2=Processed_everything2['water_frac']
    water_frac = np.concatenate([water_frac,water_frac2])
    Ocean_depth2=Processed_everything2['Ocean_depth']
    Ocean_depth = np.concatenate([Ocean_depth,Ocean_depth2])
    Max_depth2=Processed_everything2['Max_depth']
    Max_depth = np.concatenate([Max_depth,Max_depth2])
    Ocean_fraction2=Processed_everything2['Ocean_fraction']
    Ocean_fraction = np.concatenate([Ocean_fraction,Ocean_fraction2])
    TOA_L2=Processed_everything2['TOA_L']
    TOA_L = np.concatenate([TOA_L,TOA_L2])
    Plate_velocity2=Processed_everything2['Plate_velocity']
    Plate_velocity = np.concatenate([Plate_velocity,Plate_velocity2])
    Melt_volume2=Processed_everything2['Melt_volume']
    Melt_volume = np.concatenate([Melt_volume,Melt_volume2])
    O2_consumption_new2=Processed_everything2['O2_consumption_new']
    O2_consumption_new = np.concatenate([O2_consumption_new,O2_consumption_new2])
    actual_phi_surf_melt_ar2=Processed_everything2['actual_phi_surf_melt_ar']
    actual_phi_surf_melt_ar = np.concatenate([actual_phi_surf_melt_ar,actual_phi_surf_melt_ar2])
    f_O2_mantle2=Processed_everything2['f_O2_mantle']
    f_O2_mantle = np.concatenate([f_O2_mantle,f_O2_mantle2])
    iron_ratio_norm2=Processed_everything2['iron_ratio_norm']
    iron_ratio_norm = np.concatenate([iron_ratio_norm,iron_ratio_norm2])
    iron_ratio2=Processed_everything2['iron_ratio']
    iron_ratio = np.concatenate([iron_ratio,iron_ratio2])
    f_O2_MH2=Processed_everything2['f_O2_MH']
    f_O2_MH = np.concatenate([f_O2_MH,f_O2_MH2])
    f_O2_IW2=Processed_everything2['f_O2_IW']
    f_O2_IW = np.concatenate([f_O2_IW,f_O2_IW2])
    f_O2_FMQ2=Processed_everything2['f_O2_FMQ']
    f_O2_FMQ = np.concatenate([f_O2_FMQ,f_O2_FMQ2])
    mantle_H2O_fraction2=Processed_everything2['mantle_H2O_fraction']
    mantle_H2O_fraction = np.concatenate([mantle_H2O_fraction,mantle_H2O_fraction2])
    mantle_CO2_fraction2=Processed_everything2['mantle_CO2_fraction']
    mantle_CO2_fraction = np.concatenate([mantle_CO2_fraction,mantle_CO2_fraction2])
    integrate_XUV2=Processed_everything2['integrate_XUV']
    integrate_XUV = np.concatenate([integrate_XUV,integrate_XUV2])


Mp = 5.972e24*inputs_for_MC[0][1].ME
rp = 6371000*inputs_for_MC[0][1].RE
G = 6.67e-11 #gravitational constant
g = G*Mp/(rp**2) # gravity (m/s2)
rc = inputs_for_MC[0][1].rc
x_low = 1.0 
x_high =np.max(total_time[0])+0.5e9 
pm = 4000.0

mantle_mass = (4./3. * np.pi * pm * (rp**3 - rc**3))


initial_number_runs = np.shape(total_y[:,0,0])[0]
indices_wanted = []


#import pdb
#pdb.set_trace()
fin_index = -1 #-17 is 5.4 Gyrs, -13 is 6 Gyrs, -20 is 5 Gyrs, 10 is ~6.4 Gyrs
x_high = np.max(total_time[0,0:fin_index])

for k in range(0,initial_number_runs):
    mp = 5.972e24*inputs_for_MC[k][1].ME
    initCO2 = total_y[k][13][0] + total_y[k][12][0]
    initH2O = total_y[k][0][0] + total_y[k][1][0]


    #Nominal model modern volatile inventories
    Mass_fraction_volatile_surface_inclO2 = (Mass_O_atm[k][fin_index] + total_y[k][1][fin_index] - MH2O_liq[k][fin_index] + total_y[k][12][fin_index]  - MCO2_liq[k][fin_index])/mp
    #if (Mass_fraction_volatile_surface_inclO2 < 0.0013): #0.0005: #1b nominal
    #if (Mass_fraction_volatile_surface_inclO2 < 0.0008): #0.0003: #1c nominal
    # 1d is a special case (see below)
    if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.116): #1e nominal
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.14): #1f nominal
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.16): #1g nominal
    
    
    ## Smaller iron core <32.5%, tighter Agol et al. constraints on volatile envelope
    #Mass_fraction_volatile_surface_inclO2 = (Mass_O_atm[k][fin_index] + total_y[k][1][fin_index] - MH2O_liq[k][fin_index] + total_y[k][12][fin_index]  - MCO2_liq[k][fin_index])/mp
    #if (Mass_fraction_volatile_surface_inclO2 < 0.00001): #1b
    #if (Mass_fraction_volatile_surface_inclO2 < 0.00001): #1c
    # 1d is a special case (see below)
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.046):#e nominal
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.063):#f nominal
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.084):#g nominal

    ## Venus like dry oxidation efficiency and small-ish iron core
    #dry_ox = inputs_for_MC[k][5].interiorc
    #if (Mass_fraction_volatile_surface_inclO2 < 0.00001)and(dry_ox>1e-3): #1b
    #if (Mass_fraction_volatile_surface_inclO2 < 0.00001)and(dry_ox>1e-3): #1c
    # 1d is a special case (see below)
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.05)and(dry_ox>1e-3): #others
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.046)and(dry_ox>1e-3):#e nominal
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.063)and(dry_ox>1e-3):#f nominal
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.084)and(dry_ox>1e-3):#g nominal

    ## Earth-like initial volatile inventories 
    #if (Mass_fraction_volatile_surface_inclO2 < 0.00001)and(initH2O<1.4e22)and(initH2O>initCO2): #1b
    #if (Mass_fraction_volatile_surface_inclO2 < 0.00001)and(initH2O<1.4e22)and(initH2O>initCO2): #1c
    # 1d is a special case (see below)
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.046)and(initH2O<1.4e22)and(initH2O>initCO2):#e nominal
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.063)and(initH2O<1.4e22)and(initH2O>initCO2):#f nominal
    #if (((total_y[k][1][fin_index]+total_y[k][12][fin_index])/mp)< 0.084)and(initH2O<1.4e22)and(initH2O>initCO2):#g nominal

    #1d is a special case 
    #atmo_CO2_mass = total_y[k][23][fin_index]*4*3.14*rp**2/g
    #Mass_fraction_volatile_surface_inclO2 = (Mass_O_atm[k][fin_index] + water_frac[k][fin_index]*total_y[k][1][fin_index] - MH2O_liq[k][fin_index] + atmo_CO2_mass  - MCO2_liq[k][fin_index])/mp
    #if Mass_fraction_volatile_surface_inclO2<0.00004:
    #if (Mass_fraction_volatile_surface_inclO2 < 0.00001):
    #dry_ox = inputs_for_MC[k][5].interiorc
    #if (Mass_fraction_volatile_surface_inclO2 < 0.00001)and(dry_ox>1e-3):
    #if (Mass_fraction_volatile_surface_inclO2 < 0.00001)and(initH2O<1.4e22)and(initH2O>initCO2): 

    #isolating high core mass fraction
    #if core mass fraction >=25% (but <50%), then implication for outer planets is as follows:
    #if (((total_y[k][1][fin_index])/mp)< 0.116)and(((total_y[k][1][fin_index])/mp)> 0.01132)and(initH2O>initCO2): #e
    #if (((total_y[k][1][fin_index])/mp)< 0.14)and(((total_y[k][1][fin_index])/mp)> 0.02414)and(initH2O>initCO2): #f 
    #if (((total_y[k][1][fin_index])/mp)< 0.16)and(((total_y[k][1][fin_index])/mp)> 0.00574 )and(initH2O>initCO2): #g
        indices_wanted.append(k)


indices_wanted = np.array(indices_wanted)


new_num_runs = np.shape(indices_wanted)[0]
inputs_for_MC=inputs_for_MC[indices_wanted,:]
Total_Fe_array=Total_Fe_array[indices_wanted]
total_time=total_time[indices_wanted,:]
total_y=total_y[indices_wanted,:,:]
FH2O_array = FH2O_array[indices_wanted,:]
FCO2_array = FCO2_array[indices_wanted,:]
MH2O_liq=MH2O_liq[indices_wanted,:]
MCO2_liq=MCO2_liq[indices_wanted,:]
Pressre_H2O=Pressre_H2O[indices_wanted,:]
MH2O_crystal=MH2O_crystal[indices_wanted,:]
CO2_Pressure_array=CO2_Pressure_array[indices_wanted,:]
fO2_array=fO2_array[indices_wanted,:]
Mass_O_atm=Mass_O_atm[indices_wanted,:]
Mass_O_dissolved=Mass_O_dissolved[indices_wanted,:]
water_frac=water_frac[indices_wanted,:]
Ocean_depth=Ocean_depth[indices_wanted,:]
Max_depth=Max_depth[indices_wanted,:]
Ocean_fraction=Ocean_fraction[indices_wanted,:]
TOA_L=TOA_L[indices_wanted,:]
Plate_velocity=Plate_velocity[indices_wanted,:]
Melt_volume=Melt_volume[indices_wanted,:]
O2_consumption_new=O2_consumption_new[indices_wanted,:]
actual_phi_surf_melt_ar=actual_phi_surf_melt_ar[indices_wanted,:]
f_O2_mantle=f_O2_mantle[indices_wanted,:]
iron_ratio_norm=iron_ratio_norm[indices_wanted,:]
iron_ratio=iron_ratio[indices_wanted,:]
f_O2_MH=f_O2_MH[indices_wanted,:]
f_O2_IW=f_O2_IW[indices_wanted,:]
f_O2_FMQ=f_O2_FMQ[indices_wanted,:]
mantle_H2O_fraction=mantle_H2O_fraction[indices_wanted,:]
mantle_CO2_fraction=mantle_CO2_fraction[indices_wanted,:]
integrate_XUV=integrate_XUV[indices_wanted]
#rp=rp

print ('new_num_runs',new_num_runs)


[int1,int2,int3] = [2.5,50,97.5] #desired confidence intervals to be plotted

import scipy.stats
confidence_y=scipy.stats.scoreatpercentile(total_y ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_FH2O = scipy.stats.scoreatpercentile(FH2O_array ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_FCO2 = scipy.stats.scoreatpercentile(FCO2_array ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_MH2O_liq = scipy.stats.scoreatpercentile(MH2O_liq ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_MCO2_liq = scipy.stats.scoreatpercentile(MCO2_liq ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_Pressre_H2O = scipy.stats.scoreatpercentile(Pressre_H2O ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_CO2_Pressure_array = scipy.stats.scoreatpercentile(CO2_Pressure_array ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_fO2_array = scipy.stats.scoreatpercentile(fO2_array ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_Mass_O_atm = scipy.stats.scoreatpercentile(Mass_O_atm ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_Mass_O_dissolved = scipy.stats.scoreatpercentile(Mass_O_dissolved ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_water_frac = scipy.stats.scoreatpercentile(water_frac ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_Ocean_depth = scipy.stats.scoreatpercentile(Ocean_depth ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_Max_depth = scipy.stats.scoreatpercentile(Max_depth ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_Ocean_fraction = scipy.stats.scoreatpercentile(Ocean_fraction ,[int1,int2,int3], interpolation_method='fraction',axis=0)

new_atmo_ar = np.array(water_frac)*np.array(Pressre_H2O)
confidence_atmo_H2O = scipy.stats.scoreatpercentile(new_atmo_ar,[int1,int2,int3], interpolation_method='fraction',axis=0)

## Plotting of results:
confidence_O2_consumption_new = scipy.stats.scoreatpercentile(O2_consumption_new ,[int1,int2,int3], interpolation_method='fraction',axis=0)


## Mantle and magma ocean redox relative to FMQ:
confidence_f_O2_FMQ = scipy.stats.scoreatpercentile(f_O2_FMQ ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_f_O2_IW = scipy.stats.scoreatpercentile(f_O2_IW ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_f_O2_MH = scipy.stats.scoreatpercentile(f_O2_MH ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_f_O2_mantle = scipy.stats.scoreatpercentile(f_O2_mantle ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_iron_ratio = scipy.stats.scoreatpercentile(iron_ratio ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_iron_ratio_norm = scipy.stats.scoreatpercentile(iron_ratio_norm ,[int1,int2,int3], interpolation_method='fraction',axis=0)

confidence_mantle_CO2_fraction = scipy.stats.scoreatpercentile(mantle_CO2_fraction ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_mantle_H2O_fraction = scipy.stats.scoreatpercentile(mantle_H2O_fraction ,[int1,int2,int3], interpolation_method='fraction',axis=0)


FIRSTFMQ = np.copy(f_O2_FMQ)*0
SECONDFMQ = np.copy(f_O2_FMQ)*0
for k in range(0,new_num_runs):
    O2_copy = np.copy(total_y[k][22]/1e5)
    O2_copy2 = np.copy(f_O2_mantle[k])
    for i in range(0,len(total_time[k])):
        if total_y[k][2][i]>=rp:
            O2_copy[i] = 0.0
        if total_y[k][2][i]<=rc:
            O2_copy2[i] = 0.0
    FIRSTFMQ[k] = np.log10(O2_copy) - np.log10(f_O2_FMQ[k])
    SECONDFMQ[k] = np.log10(O2_copy2) - np.log10(f_O2_FMQ[k])

#pdb.set_trace()
confidence_f_O2_MAGMA_relative_FMQ = scipy.stats.scoreatpercentile(FIRSTFMQ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_f_O2_relative_FMQ = scipy.stats.scoreatpercentile(SECONDFMQ ,[int1,int2,int3], interpolation_method='fraction',axis=0)

Melt_volume = 365*24*60*60*Melt_volume/1e9
confidence_melt=scipy.stats.scoreatpercentile(Melt_volume ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_velocity =  scipy.stats.scoreatpercentile(Plate_velocity ,[int1,int2,int3], interpolation_method='fraction',axis=0)

#######################################################################
#######################################################################
##### 95% confidence interval plots (Fig. 2, 3, 5, 7 in main text)

pylab.figure(figsize=(20,10))
pylab.subplot(4,3,1)
surflabel="Surface, T$_{surf}$"
pylab.loglog(total_time[0],confidence_y[1][8],'y', label=surflabel)
pylab.fill_between(total_time[0],confidence_y[0][8], confidence_y[2][8], color='orange', alpha='0.4')  
Mantlelabel="Mantle, T$_p$"
pylab.loglog(total_time[0],confidence_y[1][7],'m', label=Mantlelabel)
pylab.fill_between(total_time[0],confidence_y[0][7], confidence_y[2][7], color='magenta', alpha='0.4')  
sol_val = sol_liq(rp,g,4000,rp,0.0,0.0)
sol_val2 = sol_liq(rp,g,4000,rp,3e9,0.0)
pylab.loglog(total_time[0],0*confidence_y[1][8]+sol_val,'k--', label='Solidus (P=0)')
#pylab.semilogx(total_time[0],0*confidence_y[1][8]+sol_val2,'g--', label='Solidus (P=3 GPa)')
pylab.ylabel("Temperature (K)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

pylab.subplot(4,3,2)
pylab.semilogx(total_time[0],confidence_y[1][2]/1000.0,'k')
pylab.fill_between(total_time[0],confidence_y[0][2]/1000.0, confidence_y[2][2]/1000.0, color='grey', alpha='0.4')  
pylab.ylabel("Radius of solidification, r$_s$ (km)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])

pylab.subplot(4,3,3)
pylab.ylabel("Pressure (bar)")
O2_label = 'O$_2$'
pylab.loglog(total_time[0],confidence_y[1][22]/1e5,'m',label=O2_label)
pylab.fill_between(total_time[0],confidence_y[0][22]/1e5,confidence_y[2][22]/1e5, color='magenta', alpha='0.4') 
CO2_label = 'CO$_2$'
pylab.loglog(total_time[0],confidence_y[1][23]/1e5,'y',label=CO2_label)
pylab.fill_between(total_time[0],confidence_y[0][23]/1e5, confidence_y[2][23]/1e5, color='orange', alpha='0.4')  
H2O_label = 'H$_2$O'
pylab.loglog(total_time[0],confidence_atmo_H2O[1]/1e5,'c',label=H2O_label)
pylab.fill_between(total_time[0],confidence_atmo_H2O[0]/1e5, confidence_atmo_H2O[2]/1e5, color='cyan', alpha='0.4')  
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

pylab.subplot(4,3,4)
pylab.ylabel("Liquid water depth (km)")
pylab.semilogx(total_time[0],confidence_Max_depth[1]/1000.0,'k--',label='Max elevation land' )
pylab.semilogx(total_time[0],confidence_Ocean_depth[1]/1000.0,'m',label='Ocean depth')
pylab.fill_between(total_time[0],confidence_Ocean_depth[0]/1000.0, confidence_Ocean_depth[2]/1000.0, color='magenta', alpha='0.4')  
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

pylab.subplot(4,3,5)
pylab.loglog(total_time[0],confidence_y[1][9] , 'm' ,label = 'OLR' )
pylab.fill_between(total_time[0],confidence_y[0][9],confidence_y[2][9], color='magenta', alpha='0.4')  
pylab.loglog(total_time[0],confidence_y[1][10] , 'y' ,label = 'ASR')
pylab.fill_between(total_time[0],confidence_y[0][10],confidence_y[2][10], color='orange', alpha='0.4')  
q_interior_label = 'q$_m$'
pylab.loglog(total_time[0],confidence_y[1][11] , 'c' ,label = q_interior_label)
pylab.fill_between(total_time[0],confidence_y[0][11],confidence_y[2][11], color='cyan', alpha='0.4')
pylab.loglog(total_time[0],280+0*confidence_y[1][9] , 'k--' ,label = 'Runaway limit')
pylab.ylabel("Heat flux (W/m2)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False,ncol=2)

pylab.subplot(4,3,6)
pylab.ylabel('CO$_2$ fluxes (Tmol/yr)')
nextlabel= 'CO$_2$ Weathering'
pylab.loglog(total_time[0],-confidence_y[1][14],'y' ,label = nextlabel )
pylab.fill_between(total_time[0],-confidence_y[2][14],-confidence_y[0][14], color='orange', alpha='0.4')  
nextlabel= 'CO$_2$ Outgassing'
pylab.loglog(total_time[0],confidence_y[1][15],'m-' ,label = nextlabel ) 
pylab.fill_between(total_time[0],confidence_y[0][15],confidence_y[2][15], color='magenta', alpha='0.4')  
pylab.yscale('symlog',linthreshy = 0.001)
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False,ncol=2)


pylab.subplot(4,3,7)
pylab.ylabel('Melt production, MP (km$^3$/yr)')
pylab.loglog(total_time[0],confidence_melt[1],'k')
pylab.fill_between(total_time[0],confidence_melt[0],confidence_melt[2], color='grey', alpha='0.4')  
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])

pylab.subplot(4,3,8)
pylab.semilogx(total_time[0],confidence_f_O2_relative_FMQ[1],'y',label = 'Solid mantle' )
pylab.fill_between(total_time[0],confidence_f_O2_relative_FMQ[0],confidence_f_O2_relative_FMQ[2], color='orange', alpha='0.4') 
pylab.semilogx(total_time[0],confidence_f_O2_MAGMA_relative_FMQ[1],'m',label = 'Magma ocean' )
pylab.fill_between(total_time[0],confidence_f_O2_MAGMA_relative_FMQ[0],confidence_f_O2_MAGMA_relative_FMQ[2], color='magenta', alpha='0.4') 
ypts = np.array([ -1.7352245862884175,1.827423167848699,2.425531914893617,3.5177304964539005,0.9172576832151291])
xpts = 4.5e9 - np.array([4.027378964941569, 4.162604340567613, 4.176627712854758, 4.345909849749583,4.363939899833055])*1e9
yrpts = 0*xpts + 2.3
pylab.ylabel("Mantle oxygen fugacity ($\Delta$QFM)")
pylab.xlabel("Time (yrs)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

pylab.subplot(4,3,9)
pylab.semilogx(total_time[0],confidence_y[1][18]*365*24*60*60/(0.032*1e12),'m' ,label = 'Dry crustal')
pylab.fill_between(total_time[0],confidence_y[0][18]*365*24*60*60/(0.032*1e12),confidence_y[2][18]*365*24*60*60/(0.032*1e12), color='magenta', alpha='0.4')  
pylab.semilogx(total_time[0],confidence_y[1][19]*365*24*60*60/(0.032*1e12),'k' ,label = 'Escape')
pylab.fill_between(total_time[0],confidence_y[0][19]*365*24*60*60/(0.032*1e12),confidence_y[2][19]*365*24*60*60/(0.032*1e12), color='grey', alpha='0.4')  
pylab.semilogx(total_time[0],confidence_y[1][20]*365*24*60*60/(0.032*1e12),'cyan' ,label = 'Wet crustal')
pylab.fill_between(total_time[0],confidence_y[0][20]*365*24*60*60/(0.032*1e12),confidence_y[2][20]*365*24*60*60/(0.032*1e12), color='cyan', alpha='0.4')  
pylab.semilogx(total_time[0],confidence_y[1][21]*365*24*60*60/(0.032*1e12),'y' ,label = 'Outgassing')
pylab.fill_between(total_time[0],confidence_y[0][21]*365*24*60*60/(0.032*1e12),confidence_y[2][21]*365*24*60*60/(0.032*1e12), color='orange', alpha='0.4')  
O2_label = 'O$_2$ flux (Tmol/yr)'
pylab.ylabel(O2_label)
pylab.yscale('symlog',linthreshy = 0.01)
pylab.legend(frameon=False,loc = 3,ncol=2)
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.minorticks_on()

pylab.subplot(4,3,10)
pylab.ylabel('Water fluxes (Tmol/yr)')
nextlabel= 'H$_2$O Escape'
pylab.loglog(total_time[0],-confidence_y[1][26]*365*24*60*60/(0.018*1e12),'c' ,label = nextlabel )
pylab.fill_between(total_time[0],-confidence_y[2][26]*365*24*60*60/(0.018*1e12),-confidence_y[0][26]*365*24*60*60/(0.018*1e12), color='cyan', alpha='0.4')  
nextlabel= 'H$_2$O Ingassing'
pylab.loglog(total_time[0],-confidence_y[1][27]*365*24*60*60/(0.018*1e12),'m' ,label = nextlabel) 
pylab.fill_between(total_time[0],-confidence_y[2][27]*365*24*60*60/(0.018*1e12),-confidence_y[0][27]*365*24*60*60/(0.018*1e12), color='magenta', alpha='0.4')  
nextlabel= 'H$_2$O Outgassing'
pylab.loglog(total_time[0],confidence_y[1][28]*365*24*60*60/(0.018*1e12),'y' ,label = nextlabel) 
pylab.fill_between(total_time[0],confidence_y[0][28]*365*24*60*60/(0.018*1e12),confidence_y[2][28]*365*24*60*60/(0.018*1e12), color='orange', alpha='0.4') 
pylab.yscale('symlog',linthreshy = 0.001)
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False,ncol=2)

pylab.subplot(4,3,11)
H2Olabel = 'H$_2$O'
pylab.semilogx(total_time[0],100*confidence_mantle_H2O_fraction[1],'y',label=H2Olabel)
pylab.fill_between(total_time[0],100*confidence_mantle_H2O_fraction[0],100*confidence_mantle_H2O_fraction[2], color='orange', alpha='0.4') 
CO2label = 'CO$_2$'
pylab.semilogx(total_time[0],100*confidence_mantle_CO2_fraction[1],'m',label=CO2label)
pylab.fill_between(total_time[0],100*confidence_mantle_CO2_fraction[0],100*confidence_mantle_CO2_fraction[2], color='magenta', alpha='0.4')  
pylab.ylabel('Solid/(Solid+Fluid) (%)')
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.minorticks_on()
pylab.legend(frameon=False)

pylab.subplot(4,3,12)
pylab.loglog(total_time[0],confidence_y[1][1],'k' ,label = 'Fluid H$_2$O')
pylab.fill_between(total_time[0],confidence_y[0][1],confidence_y[2][1], color='grey', alpha='0.4')  
pylab.loglog(total_time[0],confidence_y[1][0],'c' ,label = 'Solid mantle H$_2$O')
pylab.fill_between(total_time[0],confidence_y[0][0],confidence_y[2][0], color='cyan', alpha='0.4') 
pylab.loglog(total_time[0],confidence_y[1][12],'m' ,label = 'Fluid CO$_2$')
pylab.fill_between(total_time[0],confidence_y[0][12],confidence_y[2][12], color='magenta', alpha='0.4')  
pylab.loglog(total_time[0],confidence_y[1][13],'y' ,label = 'Solid mantle CO$_2$')
pylab.fill_between(total_time[0],confidence_y[0][13],confidence_y[2][13], color='orange', alpha='0.4')  
O2_label = 'Volatile reservoir (kg)'
pylab.ylabel(O2_label)
pylab.xlabel('Time (yrs)')
#pylab.yscale('symlog',linthreshy = 0.01)
pylab.legend(frameon=False,loc = 3,ncol=2)
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.minorticks_on()


two_by_two = "y"
if two_by_two == "y":
	pylab.figure(figsize=(20,10))
	pylab.subplot(2,2,1)
	Mantlelabel="Mantle, T$_p$"
	pylab.semilogx(total_time[0],confidence_y[1][7],'b', label=Mantlelabel)
	pylab.fill_between(total_time[0],confidence_y[0][7], confidence_y[2][7], color='blue', alpha='0.4')  
	surflabel="Surface, T$_{surf}$"
	pylab.semilogx(total_time[0],confidence_y[1][8],'r', label=surflabel)
	pylab.fill_between(total_time[0],confidence_y[0][8], confidence_y[2][8], color='red', alpha='0.4')  
	sol_val = sol_liq(rp,g,4000,rp,0.0,0.0)
	sol_val2 = sol_liq(rp,g,4000,rp,3e9,0.0)
	pylab.semilogx(total_time[0],0*confidence_y[1][8]+sol_val,'c--', label='Solidus (P=0)')
	#pylab.semilogx(total_time[0],0*confidence_y[1][8]+sol_val2,'g--', label='Solidus (P=3 GPa)')
	pylab.ylabel("Temperature (K)")
	pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
	pylab.xlim([x_low, x_high])
	pylab.legend(frameon=False)

	pylab.subplot(2,2,2)
	pylab.semilogx(total_time[0],confidence_y[1][2]/1000.0,'k')
	pylab.fill_between(total_time[0],confidence_y[0][2]/1000.0, confidence_y[2][2]/1000.0, color='grey', alpha='0.4')  
	pylab.ylabel("Radius of solidification, r$_s$ (km)")
	pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
	pylab.xlim([x_low, x_high])

	pylab.subplot(2,2,3)
	pylab.ylabel("Pressure (bar)")
	O2_label = 'O$_2$'
	pylab.loglog(total_time[0],confidence_y[1][22]/1e5,'b',label=O2_label)
	pylab.fill_between(total_time[0],confidence_y[0][22]/1e5,confidence_y[2][22]/1e5, color='blue', alpha='0.4') 
	H2O_label = 'H$_2$O'
	pylab.loglog(total_time[0],confidence_atmo_H2O[1]/1e5,'r',label=H2O_label)
	pylab.fill_between(total_time[0],confidence_atmo_H2O[0]/1e5, confidence_atmo_H2O[2]/1e5, color='red', alpha='0.4')  
	CO2_label = 'CO$_2$'
	pylab.loglog(total_time[0],confidence_y[1][23]/1e5,'k',label=CO2_label)
	pylab.fill_between(total_time[0],confidence_y[0][23]/1e5, confidence_y[2][23]/1e5, color='grey', alpha='0.6')  
	pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
	pylab.xlim([x_low, x_high])
	pylab.legend(frameon=False)

	pylab.subplot(2,2,4)
	pylab.loglog(total_time[0],confidence_y[1][9] , 'b' ,label = 'OLR' )
	pylab.fill_between(total_time[0],confidence_y[0][9],confidence_y[2][9], color='blue', alpha='0.4')  
	pylab.loglog(total_time[0],confidence_y[1][10] , 'r' ,label = 'ASR')
	pylab.fill_between(total_time[0],confidence_y[0][10],confidence_y[2][10], color='red', alpha='0.4')  
	q_interior_label = 'q$_m$'
	pylab.loglog(total_time[0],confidence_y[1][11] , 'k' ,label = q_interior_label)
	pylab.fill_between(total_time[0],confidence_y[0][11],confidence_y[2][11], color='grey', alpha='0.6')
	pylab.loglog(total_time[0],280+0*confidence_y[1][9] , 'c--' ,label = 'Runaway limit')
	pylab.ylabel("Heat flux (W/m2)")
	pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
	pylab.xlim([x_low, x_high])
	pylab.legend(frameon=False,ncol=2)


if two_by_two == "y":
	pylab.figure(figsize=(20,10))
	pylab.subplot(4,2,1)
	Mantlelabel="Mantle, T$_p$"
	pylab.semilogx(total_time[0],confidence_y[1][7],'b', label=Mantlelabel)
	pylab.fill_between(total_time[0],confidence_y[0][7], confidence_y[2][7], color='blue', alpha='0.4')  
	surflabel="Surface, T$_{surf}$"
	pylab.semilogx(total_time[0],confidence_y[1][8],'r', label=surflabel)
	pylab.fill_between(total_time[0],confidence_y[0][8], confidence_y[2][8], color='red', alpha='0.4')  
	sol_val = sol_liq(rp,g,4000,rp,0.0,0.0)
	sol_val2 = sol_liq(rp,g,4000,rp,3e9,0.0)
	pylab.semilogx(total_time[0],0*confidence_y[1][8]+sol_val,'c--', label='Solidus (P=0)')
	#pylab.semilogx(total_time[0],0*confidence_y[1][8]+sol_val2,'g--', label='Solidus (P=3 GPa)')
	pylab.ylabel("Temperature (K)")
	pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
	pylab.xlim([x_low, x_high])
	pylab.legend(frameon=False)

	pylab.subplot(4,2,2)
	pylab.semilogx(total_time[0],confidence_y[1][2]/1000.0,'k')
	pylab.fill_between(total_time[0],confidence_y[0][2]/1000.0, confidence_y[2][2]/1000.0, color='grey', alpha='0.4')  
	pylab.ylabel("Radius of solidification, r$_s$ (km)")
	pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
	pylab.xlim([x_low, x_high])

	pylab.subplot(4,2,3)
	pylab.ylabel("Pressure (bar)")
	O2_label = 'O$_2$'
	pylab.loglog(total_time[0],confidence_y[1][22]/1e5,'b',label=O2_label)
	pylab.fill_between(total_time[0],confidence_y[0][22]/1e5,confidence_y[2][22]/1e5, color='blue', alpha='0.4') 
	H2O_label = 'H$_2$O'
	pylab.loglog(total_time[0],confidence_atmo_H2O[1]/1e5,'r',label=H2O_label)
	pylab.fill_between(total_time[0],confidence_atmo_H2O[0]/1e5, confidence_atmo_H2O[2]/1e5, color='red', alpha='0.4')  
	CO2_label = 'CO$_2$'
	pylab.loglog(total_time[0],confidence_y[1][23]/1e5,'k',label=CO2_label)
	pylab.fill_between(total_time[0],confidence_y[0][23]/1e5, confidence_y[2][23]/1e5, color='grey', alpha='0.6')  
	pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
	pylab.xlim([x_low, x_high])
	pylab.legend(frameon=False)

	pylab.subplot(4,2,4)
	pylab.loglog(total_time[0],confidence_y[1][9] , 'b' ,label = 'OLR' )
	pylab.fill_between(total_time[0],confidence_y[0][9],confidence_y[2][9], color='blue', alpha='0.4')  
	pylab.loglog(total_time[0],confidence_y[1][10] , 'r' ,label = 'ASR')
	pylab.fill_between(total_time[0],confidence_y[0][10],confidence_y[2][10], color='red', alpha='0.4')  
	q_interior_label = 'q$_m$'
	pylab.loglog(total_time[0],confidence_y[1][11] , 'k' ,label = q_interior_label)
	pylab.fill_between(total_time[0],confidence_y[0][11],confidence_y[2][11], color='grey', alpha='0.6')
	pylab.loglog(total_time[0],280+0*confidence_y[1][9] , 'c--' ,label = 'Runaway limit')
	pylab.ylabel("Heat flux (W/m2)")
	pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
	pylab.xlim([x_low, x_high])
	pylab.legend(frameon=False,ncol=2)

	pylab.subplot(4,2,5)
	pylab.loglog(total_time[0],confidence_y[1][29],'k')
	pylab.fill_between(total_time[0],confidence_y[0][29], confidence_y[2][29], color='grey', alpha='0.4')  
	pylab.ylabel("fH2O stratosphere")
	pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
	pylab.xlim([x_low, x_high])

	pylab.subplot(4,2,6)
	pylab.semilogx(total_time[0],confidence_y[1][30],'k')
	pylab.fill_between(total_time[0],confidence_y[0][30], confidence_y[2][30], color='grey', alpha='0.4')  
	pylab.ylabel("Upper atmo temp (K)")
	pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
	pylab.xlim([x_low, x_high])

	pylab.subplot(4,2,7)
	pylab.loglog(total_time[0],confidence_y[1][31],'k')
	pylab.fill_between(total_time[0],confidence_y[0][31], confidence_y[2][31], color='grey', alpha='0.4')  
	pylab.ylabel("Atmo H2O (Pa)")
	pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
	pylab.xlim([x_low, x_high])


##### Done with 3x3 confidence interval plot
############################################


pylab.figure()
CO2label = 'CO$_2$'
pylab.semilogx(total_time[0],confidence_mantle_CO2_fraction[1],'g',label=CO2label)
pylab.fill_between(total_time[0],confidence_mantle_CO2_fraction[0],confidence_mantle_CO2_fraction[2], color='green', alpha='0.4')  
pylab.ylabel('Fraction solid')
#pylab.xlabel('Time (yrs)')
H2Olabel = 'H$_2$O'
pylab.semilogx(total_time[0],confidence_mantle_H2O_fraction[1],'b',label=H2Olabel)
pylab.fill_between(total_time[0],confidence_mantle_H2O_fraction[0],confidence_mantle_H2O_fraction[2], color='blue', alpha='0.4') 
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.minorticks_on()
pylab.legend(frameon=False)

############################################
pylab.figure()
pylab.ylabel("MMW")
pylab.semilogx(total_time[0],confidence_y[1][24],'g')
pylab.fill_between(total_time[0],confidence_y[0][24],confidence_y[2][24], color='green', alpha='0.4')  
pylab.xlabel('Time (yrs)')

pylab.figure()
pylab.subplot(2,1,1)
pylab.ylabel("Solid mantle Fe3+/Fe2+")
pylab.semilogx(total_time[0],confidence_iron_ratio[1],'k')
pylab.fill_between(total_time[0],confidence_iron_ratio[0],confidence_iron_ratio[2], color='grey', alpha='0.4')  
pylab.xlabel('Time (yrs)')
pylab.subplot(2,1,2)
pylab.ylabel("Solid mantle Fe3+/FeTot")
pylab.semilogx(total_time[0],confidence_iron_ratio_norm[1],'k')
pylab.fill_between(total_time[0],confidence_iron_ratio_norm[0],confidence_iron_ratio_norm[2], color='grey', alpha='0.4')  
pylab.xlabel('Time (yrs)')
pylab.tight_layout()

#######################################################################
#######################################################################
### Begin parameter space plots (e.g. Fig. 4, 6 and Fig. S1)
######## First, need to fill input arrays from input files
#######################################################################

pylab.figure()
pylab.subplot(4,1,1)
O2_final_ar = []
H2O_final_ar = []
CO2_final_ar = []
Total_P_ar = []
atmo_H2O_ar = []
Hab_counter = 0
Hab_counter_anox = 0
TotalP_counter=0
TotalP_atmo_counter = 0

for k in range(0,new_num_runs):
    addO2 = total_y[k][22][fin_index]/1e5 
    if (addO2 <= 0 ) or np.isnan(addO2):
        addO2 = 1e-8
    addH2O = Pressre_H2O[k][fin_index]/1e5
    atmo_H2O = water_frac[k][fin_index]*Pressre_H2O[k][fin_index]/1e5 
    if (addH2O <= 0 ) or np.isnan(addH2O):
        addH2O = 1e-8
    if (atmo_H2O<=0) or np.isnan(atmo_H2O):
        atmo_H2O = 0
    addCO2 = total_y[k][23][fin_index]/1e5 #CO2_Pressure_array[k][fin_index]/1e5
    if (addCO2 <= 0 ) or np.isnan(addCO2):
        addCO2 = 1e-8
    O2_final_ar.append(np.log10(addO2))
    H2O_final_ar.append(np.log10(addH2O))
    CO2_final_ar.append(np.log10(addCO2))
    atmo_H2O_ar.append(np.log10(atmo_H2O))
    #atmo_H2O_ar.append(atmo_H2O)
    Total_P_ar.append(np.log10(addO2+addH2O+addCO2))
    if addO2+addH2O+addCO2>1.0:
        TotalP_counter = TotalP_counter + 1
    if addO2+atmo_H2O+addCO2>1.0:
        TotalP_atmo_counter = TotalP_atmo_counter + 1
    if (total_y[k][8][fin_index]>250.0)and( water_frac[k][fin_index]<1):
        Hab_counter = Hab_counter+1
    if (total_y[k][8][fin_index]>250.0)and( water_frac[k][fin_index]<1)and(addO2<0.01):
        Hab_counter_anox = Hab_counter_anox+1

print ('counters',TotalP_counter,Hab_counter,TotalP_atmo_counter,Hab_counter_anox,'out of',new_num_runs)
print ('counters',TotalP_counter/new_num_runs,Hab_counter/new_num_runs,TotalP_atmo_counter/new_num_runs,Hab_counter_anox/new_num_runs)


pylab.hist(O2_final_ar,bins = 50,color = 'g')
pylab.xlabel('log(pO2)')
pylab.subplot(4,1,2)
pylab.hist(H2O_final_ar,color = 'b',bins = 50)
pylab.xlabel('log(pH2O)')
pylab.subplot(4,1,3)
pylab.hist(CO2_final_ar,color = 'r',bins = 50)
pylab.xlabel('log(pCO2)')
pylab.subplot(4,1,4)
pylab.hist(Total_P_ar,color = 'c',bins = 50)
pylab.xlabel('Log10(Pressure (bar))')

init_CO2_H2O = []
init_CO2_ar =[]
Final_O2 = []
Final_CO2 = []
Final_H2O = []
Surface_T_ar = []
H2O_upper_ar = []
Weathering_limit = []
tsat_array = []
beta_XUV_array=[]
fsat_XUV_array = []
epsilon_array = []

Ca_array = []
Omega_ar = []
init_H2O_ar=[]
offset_ar = []
Te_ar =[]
expCO2_ar = []
Mfrac_hydrated_ar= []
dry_frac_ar = []
wet_OxFrac_ar = []
Radiogenic= []
Init_fluid_O_ar = []
albedoH_ar = []
albedoC_ar = []
MaxMantleH2O=[]
imp_coef_ar = []
imp_slope_ar = []
hist_total_imp_mass=[]
mult_ar= []
mix_epsilon_ar=[]
Tstrat_ar = []
Tstrat_true_ar = []
fH2O_start = []
surface_magma_frac_array = []
ThermoTemp_ar = []
iron_ratio_norm_final = []
#Final_CO2og = []
Pressre_H2O_ar=[]

for k in range(0,new_num_runs):
    Tstrat_ar.append(inputs_for_MC[k][5].Tstrat)
    Tstrat_true_ar.append(total_y[k][30][fin_index])
    surface_magma_frac_array.append(inputs_for_MC[k][5].surface_magma_frac)
    iron_ratio_norm_final.append(iron_ratio_norm[k][fin_index])
    init_CO2 = total_y[k][12][0]+total_y[k][13][0]
    init_H2O = total_y[k][0][0]+total_y[k][1][0]
    init_H2O_ar.append(inputs_for_MC[k][2].Init_fluid_H2O)
    Init_fluid_O_ar.append(inputs_for_MC[k][2].Init_fluid_O)
    albedoC_ar.append(inputs_for_MC[k][1].albedoC)
    albedoH_ar.append(inputs_for_MC[k][1].albedoH)
    init_CO2_H2O.append(init_CO2/init_H2O)
    init_CO2_ar.append(inputs_for_MC[k][2].Init_fluid_CO2)
    #Final_O2.append( Mass_O_atm[k][fin_index]*g/(4*np.pi*(0.032/total_y[k][24][fin_index])*rp**2*1e5))
    if total_y[k][22][fin_index]/1e5 < 0.9e-6:#1.2e-6: ## make sure correlations aren't messed up by cutoff
        Final_O2.append( 1e-6 )
        #Final_O2.append( 1e-10 )
    else:
        Final_O2.append( total_y[k][22][fin_index]/1e5)
    #Final_CO2og.append(CO2_Pressure_array[k][fin_index]/1e5) 
    Final_CO2.append(total_y[k][23][fin_index]/1e5) 
    Final_H2O.append(water_frac[k][fin_index]*Pressre_H2O[k][fin_index]/1e5)
    Pressre_H2O_ar.append(Pressre_H2O[k][fin_index]/1e5)
    H2O_upper_ar.append(total_y[k][14][fin_index])
    Surface_T_ar.append(total_y[k][8][fin_index])
    fH2O_start.append(total_y[k][29][fin_index])
    Weathering_limit.append(inputs_for_MC[k][5].supp_lim)
    tsat_array.append(inputs_for_MC[k][4].tsat_XUV)
    beta_XUV_array.append(inputs_for_MC[k][4].beta0)
    fsat_XUV_array.append(inputs_for_MC[k][4].fsat)
    epsilon_array.append(inputs_for_MC[k][4].epsilon)
    Ca_array.append(inputs_for_MC[k][5].ocean_a)
    Omega_ar.append(inputs_for_MC[k][5].ocean_b)
    offset_ar.append(inputs_for_MC[k][5].interiora)
    Mfrac_hydrated_ar.append(inputs_for_MC[k][5].interiorb)
    Te_ar.append(inputs_for_MC[k][5].ccycle_a)
    expCO2_ar.append(inputs_for_MC[k][5].ccycle_b) 
    dry_frac_ar.append(inputs_for_MC[k][5].interiorc)
    wet_OxFrac_ar.append(inputs_for_MC[k][5].interiord)
    Radiogenic.append(inputs_for_MC[k][5].interiore)
    MaxMantleH2O.append(inputs_for_MC[k][5].interiorf)
    imp_coef_ar.append(inputs_for_MC[k][5].esc_a)
    imp_slope_ar.append(inputs_for_MC[k][5].esc_b)
    mult_ar.append(inputs_for_MC[k][5].esc_c)
    mix_epsilon_ar.append(inputs_for_MC[k][5].esc_d)
    ThermoTemp_ar.append(inputs_for_MC[k][5].ThermoTemp)
    t_ar = np.linspace(0,1,1000)
    y = np.copy(t_ar)
    for i in range(0,len(t_ar)):
        y[i] = inputs_for_MC[k][5].esc_a*np.exp(-t_ar[i]/inputs_for_MC[k][5].esc_b)
    hist_total_imp_mass.append(np.trapz(y,t_ar*1e9))



####################!!!!!!
####################!!!!!!
####################!!!!!!
####################!!!!!!
## optional for redoing correlation atmoCO2
#Final_O2 = np.copy(Final_CO2) 
#for k in range(0,new_num_runs):
#    if Final_O2[k]<0:
#        Final_O2[k] = 1e-5
## optional for redoing correlation atmoH2O
#Final_O2 = np.copy(Final_H2O)
#for k in range(0,new_num_runs):
#    if Final_O2[k]<1e-3:
#        Final_O2[k] = 1e-3
## optional for redoing correlation Total P
#Final_O2 = np.copy(Pressre_H2O_ar)+np.copy(Final_CO2)+np.copy(Final_O2)
#for k in range(0,new_num_runs):
#    if Final_O2[k]<1e-3:
#        Final_O2[k] = 1e-3
####################!!!!!!
####################!!!!!!
####################!!!!!!
####################!!!!!!


Ca_array = np.array(Ca_array)
Omega_ar = np.array(Omega_ar)


correlation_array=[]
pval_array=[]
variabl_array=[]


pylab.figure()

pylab.subplot(2,2,1)
pylab.loglog(init_H2O_ar,Final_O2,'.')
pylab.xlabel('init_H2O')
pylab.ylabel('Final O2 (bar)')
xxx = np.log10(init_H2O_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('init_H2O')

pylab.subplot(2,2,2)
pylab.loglog(init_CO2_ar,Final_O2,'.')
pylab.ylabel('Final_O2')
pylab.xlabel('Initial CO2 inventory')
xxx = np.log10(init_CO2_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('Initial CO2 inventory')

pylab.subplot(2,2,3)
pylab.loglog(Radiogenic,Final_O2,'.')
pylab.xlabel('Radiogenic')
pylab.ylabel('Final O2 (bar)')
xxx = np.log10(Radiogenic)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('Radiogenic')

pylab.subplot(2,2,4)
pylab.loglog(Init_fluid_O_ar,Final_O2,'.')
pylab.xlabel('Init_fluid_O')
pylab.ylabel('Final_O2')
xxx = np.log10(Init_fluid_O_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('Init_fluid_O')


pylab.figure()
pylab.subplot(3,3,1)
pylab.loglog(tsat_array,Final_O2,'.')
pylab.xlabel('tsat XUV (Gyr)')
pylab.ylabel('final pO2 (bar)')
xxx = np.log10(tsat_array)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('tsat XUV (Gyr)')


pylab.subplot(3,3,2)
pylab.semilogy(beta_XUV_array,Final_O2,'.')
pylab.xlabel('beta_XUV_array')
pylab.ylabel('final pO2 (bar)')
xxx =  np.array(beta_XUV_array)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.semilogy(xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('beta_XUV_array')

pylab.subplot(3,3,3)
pylab.loglog(fsat_XUV_array,Final_O2,'.')
pylab.xlabel('fsat_XUV')
pylab.ylabel('final pO2 (bar)')
xxx = np.log10(fsat_XUV_array)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('fsat_XUV')

pylab.subplot(3,3,4)
pylab.loglog(integrate_XUV,Final_O2,'s',color='r',label='Final O2')
pylab.xlabel('integrate_XUV')
pylab.ylabel('final pO2 (bar)')
xxx = np.log10(integrate_XUV)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('integrate_XUV')

pylab.subplot(3,3,5)
pylab.semilogy(epsilon_array,Final_O2,'.')
pylab.xlabel('Low XUV escape efficiency, $\epsilon$$_{lowXUV}$ ')
pylab.ylabel('Final O$_2$ (bar)')
xxx = np.array(epsilon_array)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.semilogy(xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('Low XUV escape efficiency, $\epsilon$$_{lowXUV}$')

pylab.subplot(3,3,6)
pylab.loglog(mult_ar,Final_O2,'.')
pylab.xlabel('mult_ar')
pylab.ylabel('Final O2 (bar)')
xxx = np.log10(mult_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('mult_ar')

pylab.subplot(3,3,7)
pylab.semilogy(mix_epsilon_ar,Final_O2,'.')
pylab.xlabel('mix_epsilon_ar')
pylab.ylabel('Final O2 (bar)')
xxx = np.array(mix_epsilon_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.semilogy(xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('mix_epsilon_ar')

pylab.subplot(3,3,8)
pylab.semilogy(Tstrat_ar,Final_O2,'b.') 
pylab.xlabel('Delta T$_{stratosphere}$ (K)')
pylab.ylabel('Final O$_2$ (bar)')
#pylab.xlim([150,250])
xxx = np.array(Tstrat_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.semilogy(xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('T$_{stratosphere}$ (K)')

pylab.subplot(3,3,9)
pylab.loglog(ThermoTemp_ar,Final_O2,'.')
pylab.xlabel('ThermoTemp_ar (K)')
pylab.ylabel('final pO2 (bar)')
xxx = np.log10(ThermoTemp_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('ThermoTemp_ar (K)')

pylab.figure()
pylab.subplot(2,3,5)
pylab.loglog(imp_coef_ar,Final_O2,'.')
pylab.xlabel('NTloss')
pylab.ylabel('Final O2 (bar)')
xxx = np.log10(imp_coef_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('NTloss')

pylab.subplot(2,3,1)
pylab.semilogy(Te_ar,Final_O2,'.')
pylab.xlabel('Te_ar (K)')
pylab.ylabel('Final O2 (bar)')
xxx = np.array(Te_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.semilogy(xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('Te_ar (K)')

pylab.subplot(2,3,2)
pylab.semilogy(expCO2_ar,Final_O2,'.')
pylab.xlabel('expCO2_ar')
pylab.ylabel('Final O2 (bar)')
xxx = np.array(expCO2_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.semilogy(xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('expCO2_ar')

pylab.subplot(2,3,3)
pylab.loglog(Weathering_limit,Final_O2,'.')
pylab.xlabel('Weathering limit (kg/s)')
pylab.ylabel('final pO2 (bar)')
xxx = np.log10(Weathering_limit)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('Weathering limit (kg/s)')

pylab.subplot(2,3,4)
pylab.loglog(Omega_ar/Ca_array,Final_O2,'.')
pylab.xlabel('omega/Ca ~ CO3')
pylab.ylabel('Final O2 (bar)')
xxx = np.log10(Omega_ar/Ca_array)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('omega/Ca ~ CO3')

pylab.figure()
pylab.subplot(2,3,1)
pylab.loglog(offset_ar,Final_O2,'.')
pylab.xlabel('offset')
pylab.ylabel('Final O2 (bar)')
xxx = np.log10(offset_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('offset')

pylab.subplot(2,3,2)
pylab.loglog(Mfrac_hydrated_ar,Final_O2,'.')
pylab.xlabel('Mfrac_hydrated_ar')
pylab.ylabel('Final O2 (bar)')
xxx = np.log10(Mfrac_hydrated_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('Mfrac_hydrated_ar')

pylab.subplot(2,3,3)
pylab.loglog(dry_frac_ar,Final_O2,'.')
pylab.xlabel('Efficiency dry crustal oxidation, $f_{dry-oxid}$')
pylab.ylabel('Final O$_2$ (bar)')
xxx = np.log10(dry_frac_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('Efficiency dry crustal oxidation, $f_{dry-oxid}$')

pylab.subplot(2,3,4)
pylab.loglog(wet_OxFrac_ar,Final_O2,'.')
pylab.xlabel('wet_OxFrac_ar')
pylab.ylabel('Final O2 (bar)')
xxx = np.log10(wet_OxFrac_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('wet_OxFrac_ar')

pylab.subplot(2,3,5)
pylab.loglog(surface_magma_frac_array,Final_O2,'.')
pylab.xlabel('surface_magma_frac_array')
pylab.ylabel('Final O$_2$ (bar)')
xxx = np.log10(surface_magma_frac_array)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('surface_magma_frac_array')

pylab.subplot(2,3,6)
pylab.semilogy(MaxMantleH2O,Final_O2,'.')
pylab.xlabel('MaxMantleH2O')
pylab.ylabel('Final_O2')
xxx = np.array(MaxMantleH2O)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.semilogy(xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('MaxMantleH2O')

pylab.figure()
pylab.subplot(1,2,1)
pylab.semilogy(albedoH_ar,Final_O2,'.')
pylab.xlabel('albedoH_ar')
pylab.ylabel('Final_O2')
xxx = np.array(albedoH_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.semilogy(xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('albedoH_ar')

pylab.subplot(1,2,2)
pylab.semilogy(albedoC_ar,Final_O2,'.')
pylab.xlabel('AlbedoC_ar')
pylab.ylabel('Final_O2')
xxx = np.array(albedoC_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)
pylab.title([round(result.slope,3),round(result.stderr,3),round(result.rvalue,3),round(result.pvalue,3)])
pylab.semilogy(xxx,10**(xxx*result.slope+result.intercept),'r-')

correlation_array.append(result[2])
pval_array.append(result[3])
variabl_array.append('AlbedoC_ar')





pylab.figure()
pylab.subplot(2,1,1)
pylab.loglog(init_H2O_ar,Final_O2,'.')
pylab.xlabel('Initial H$_2$O (kg)')
pylab.ylabel('Final O$_2$ (bar)')

pylab.subplot(2,1,2)
pylab.loglog(np.array(init_H2O_ar)/1.4e21,Final_O2,'.')
pylab.xlabel('Initial H$_2$O (Earth oceans)')
pylab.ylabel('Final O$_2$ (bar)')


from sigfig import round
pylab.figure(figsize=(6,16))
pylab.subplot(3,1,1)
pylab.loglog(np.array(init_H2O_ar)/1.4e21,Final_O2,'.')
pylab.xlabel('Initial H$_2$O (Earth oceans)')
pylab.ylabel('Final O$_2$ (bar)')
xxx = np.log10(np.array(init_H2O_ar)/1.4e21)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)

frs = round(result.rvalue,sigfigs=3)
sec = round(result.pvalue,sigfigs=3)
leg = 'R$^2$=%s, p=%s' %(frs, sec)
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-',label=leg)
pylab.legend()

pylab.subplot(3,1,2)
pylab.loglog(dry_frac_ar,Final_O2,'.')
pylab.xlabel('Efficiency dry crustal oxidation, $f_{dry-oxid}$')
pylab.ylabel('Final O$_2$ (bar)')
xxx = np.log10(dry_frac_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)

frs = round(result.rvalue,sigfigs=3)
sec = round(result.pvalue,sigfigs=3)
leg = 'R$^2$=%s, p=%s' %(frs, sec)
pylab.loglog(10**xxx,10**(xxx*result.slope+result.intercept),'r-',label=leg)
pylab.legend()

pylab.subplot(3,1,3)
pylab.semilogy(albedoC_ar,Final_O2,'.')
pylab.xlabel('Albedo (cold state)')
pylab.ylabel('Final O$_2$ (bar)')
xxx = np.array(albedoC_ar)
yyy=np.log10(Final_O2)
result = scipy.stats.linregress(xxx,yyy)

frs = round(result.rvalue,sigfigs=3)
sec = round(result.pvalue,sigfigs=3)
leg = 'R$^2$=%s, p=%s' %(frs, sec)
pylab.semilogy(xxx,10**(xxx*result.slope+result.intercept),'r-',label=leg)
pylab.legend()



correlation_array=np.array(correlation_array)
pval_array=np.array(pval_array)
variabl_array = np.array(variabl_array)



pylab.show()

