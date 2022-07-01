##################### 
# load modules
import time
import numpy as np
import pylab
from joblib import Parallel, delayed
from all_classes_Ledits import * 
from Main_code_callable import forward_model
import sys
import os
import shutil
import contextlib
import pdb
####################

num_runs = 720 # Number of forward model runs
num_cores = 60 # For parallelization, check number of cores with multiprocessing.cpu_count()
os.mkdir('switch_garbage3')

#Choose planet
which_planet = "Tnew" ## revised Trappist for paper


if which_planet=="Tnew":

    k_trap = 3 # 0-b, 1-c, 2-d, 3-e, 4-f, 5-g ### Choose Trappist-1 planet here
    
    TRAPPIST_inputs = Switch_Inputs(print_switch = "n", speedup_flag = "n", start_speed=11e6 , fin_speed=100e6,heating_switch = 0,C_cycle_switch="y",Start_time=10e6)   

    TRAPPIST_R_array = np.array([1.116,1.097,0.788,.920,1.045,1.129,0.775]) ## Planet radii (Agol et al.)
    TRAPPIST_M_array = np.array([1.374,1.308,0.388,.692,1.039,1.321,0.326]) ## Planet masses (Agol et al.)

    TRAPPIST_sep_array = np.array([0.01154, 0.0158,0.02227,0.02925,0.03849,0.04683,.06189 ]) # Orbital separations
    TRAPPIST_Rc_array = TRAPPIST_R_array * 3.4e6  ## Approximate metallic core radius (only affects convective heatflow)
    TRAPPIST_step1_array = np.array([10000,10000,10000,10000,10000,10000,10000]) 

    Tefold = np.random.uniform(5,30,num_runs) #e-folding temperature silicate weathering
    alphaexp = np.random.uniform(0.1,0.5,num_runs) #CO2 dependence silicate weathering
    suplim_ar = 10**np.random.uniform(5,7,num_runs) #supply limit silicate weathering

    tdc = np.random.uniform(0.06,0.14,num_runs) #not used this model version
    mult_ar = 10**np.random.uniform(-2,2,num_runs) #transition abundance diffusion-limited to XUV-limited
    mix_epsilon_ar = np.random.uniform(0.0,1.0,num_runs) #fraction energy driving loss above O-drag threshold

    init_O = 10**np.random.uniform(20.6,22,num_runs) #initial free oxygen (sets mantle redox)
    Albedo_C_range = np.random.uniform(0.01,0.5,num_runs) # Temperate albedo
    Albedo_H_range = np.random.uniform(0.0001,0.2,num_runs) #Runaway greenhouse albedo
    for k in range(0,len(Albedo_C_range)): #check albedo consistency
        if Albedo_C_range[k] < Albedo_H_range[k]:
            Albedo_H_range[k] = Albedo_C_range[k]-1e-5    
       
    Epsilon_ar = np.random.uniform(0.01,0.3,num_runs) #efficiency of XUV-driven escape at low XUV fluxes

    # Trappist-1 stellar parameters (Birky et al. distributions)
    ace = np.load('trappist_posterior_samples_updated.npy')
    indices_narrow = np.where((ace[:,3]>7) & (ace[:,3]<9)) #restrict ages to ~ 8 Gyr
    ace_narrow = ace[indices_narrow]
    stellar_sample_index = np.random.randint(0,len(ace_narrow),num_runs)
    fsat_ar = 10**ace_narrow[stellar_sample_index,1] #Saturated XUV ratio
    tsat_ar = ace_narrow[stellar_sample_index,2] #Saturation time
    beta_ar = ace_narrow[stellar_sample_index,4] #XUV decay coefficient

    ocean_Ca_ar = 10**np.random.uniform(-4,-0.52,num_runs) ## Ocean [Ca]
    ocean_Omega_ar = np.random.uniform(1.0,10.0,num_runs) ## Ocean Saturation state
    MFrac_hydrated_ar = 10**np.random.uniform(np.log10(0.001),np.log10(0.03),num_runs) #crustal hydration efficiency 

    dry_ox_frac_ac = 10**np.random.uniform(-4,-1,num_runs) #dry oxidation efficiency 
    wet_oxid_eff_ar = 10**np.random.uniform(-3,-1,num_runs) #wet oxidation efficiency 
    Mantle_H2O_max_ar = np.random.uniform(0.5,15.0,num_runs) # Maximum mantle water content

    Tstrat_array = np.random.uniform(-30.0,30.0,num_runs) ## variation of cold trap temperature about skin temperature
    surface_magma_frac_array = 10**np.random.uniform(-4,0,num_runs) #surface molten fraction 

    offset_range = 10**np.random.uniform(1.0,3.0,num_runs) ## Mantle viscosity coefficient
    heatscale_ar = 10**np.random.uniform(-0.48,1.477,num_runs) ## Internal heating (initial radiogenics), relative Earth

    Thermosphere_temp = 10**np.random.uniform(2.3,3.699,num_runs) #Johnstone papers, 200 K - 5000 K

    #new initial volatile abundances, and non thermal escape
    init_water = 10**np.random.uniform(21,23.63,num_runs) #up to 300 Earth oceans
    init_CO2 = 10**np.random.uniform(20,22.69,num_runs) ## not enough carbon to affect mantle dynamics, and limits radiation scheme
    NT_loss = 10**np.random.uniform(0.0,2.0,num_runs) ## constant non thermal loss in atmospheres, total after 8 Gyr (bar)


##Output arrays and parameter inputs to be filled:
inputs = range(0,len(init_water))
output = []

for zzz in inputs:
    ii = zzz
    
    if which_planet =="Tnew":
        TRAPPIST_Init_conditions = Init_conditions(Init_solid_H2O=0.0, Init_fluid_H2O=init_water[ii] , Init_solid_O=0.0, Init_fluid_O=TRAPPIST_M_array[k_trap]*init_O[ii], Init_solid_FeO1_5 = 0.0, Init_solid_FeO=0.0, Init_solid_CO2=0.0, Init_fluid_CO2 =  init_CO2[ii])   
        TRAPPIST_Numerics = Numerics(total_steps = 3 ,step0 = 100.0, step1=TRAPPIST_step1_array[k_trap] , step2=10000, step3=1e6, step4=-999, tfin0=1e7+10000, tfin1=100e6, tfin2=300e6, tfin3=7.96e9, tfin4 = -999)
        TRAPPIST_Planet_inputs = Planet_inputs(RE = TRAPPIST_R_array[k_trap], ME = TRAPPIST_M_array[k_trap], rc=TRAPPIST_Rc_array[k_trap], pm=4000.0, Total_Fe_mol_fraction = 0.06, Planet_sep=TRAPPIST_sep_array[k_trap],albedoC=Albedo_C_range[ii], albedoH=Albedo_H_range[ii])   
        TRAPPIST_Stellar_inputs = Stellar_inputs(tsat_XUV=tsat_ar[ii], Stellar_Mass=0.09, fsat=fsat_ar[ii], beta0=beta_ar[ii], epsilon=Epsilon_ar[ii])    
        MC_inputs_ar = MC_inputs(esc_a=NT_loss[ii], esc_b=tdc[ii],  esc_c = mult_ar[ii], esc_d = mix_epsilon_ar[ii], ccycle_a=Tefold[ii] , ccycle_b=alphaexp[ii], supp_lim =suplim_ar[ii], interiora =offset_range[ii], interiorb=MFrac_hydrated_ar[ii],interiorc=dry_ox_frac_ac[ii],interiord = wet_oxid_eff_ar[ii],interiore = heatscale_ar[ii], interiorf = Mantle_H2O_max_ar[ii], ocean_a=ocean_Ca_ar[ii],ocean_b=ocean_Omega_ar[ii],Tstrat = Tstrat_array[ii], surface_magma_frac = surface_magma_frac_array[ii],ThermoTemp = Thermosphere_temp[ii])
        inputs_for_later = [TRAPPIST_inputs,TRAPPIST_Planet_inputs,TRAPPIST_Init_conditions,TRAPPIST_Numerics,TRAPPIST_Stellar_inputs,MC_inputs_ar]
    
    sve_name = 'switch_garbage3/inputs4L%d' %ii
    np.save(sve_name,inputs_for_later)

## Attempt to run forward model three times, each with slightly different numerical solver options:
def processInput(i):
    load_name = 'switch_garbage3/inputs4L%d.npy' %i
    try:
        if which_planet == "Tnew":
            print ('starting ',i)
            [TRAPPIST_inputs,TRAPPIST_Planet_inputs,TRAPPIST_Init_conditions,TRAPPIST_Numerics,TRAPPIST_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
            TRAPPIST_Numerics.total_steps = 9 #first attempt
            outs = forward_model(TRAPPIST_inputs,TRAPPIST_Planet_inputs,TRAPPIST_Init_conditions,TRAPPIST_Numerics,TRAPPIST_Stellar_inputs,MC_inputs_ar)
            if outs.total_time[-1]<4.4e9:
                print (i,' finished early')
                abc=np.load('thisisnotreal.npy',allow_pickle=True)
    except:
        try: # try again with slightly different numerical options
            if which_planet =="Tnew":  
                print ('trying ',i,' again')
                [TRAPPIST_inputs,TRAPPIST_Planet_inputs,TRAPPIST_Init_conditions,TRAPPIST_Numerics,TRAPPIST_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
                TRAPPIST_Numerics.total_steps = 3  #second attempt
                outs = forward_model(TRAPPIST_inputs,TRAPPIST_Planet_inputs,TRAPPIST_Init_conditions,TRAPPIST_Numerics,TRAPPIST_Stellar_inputs,MC_inputs_ar)
                if outs.total_time[-1]<4.4e9:
                    print (i,' finished early')
                    abc=np.load('thisisnotreal.npy',allow_pickle=True)

        except:
            try: # try again with slightly different numerical options
                if which_planet=="Tnew": 
                    print ('trying ',i,' final time')
                    [TRAPPIST_inputs,TRAPPIST_Planet_inputs,TRAPPIST_Init_conditions,TRAPPIST_Numerics,TRAPPIST_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
                    TRAPPIST_Numerics.total_steps = 99 #third attempt

                    outs = forward_model(TRAPPIST_inputs,TRAPPIST_Planet_inputs,TRAPPIST_Init_conditions,TRAPPIST_Numerics,TRAPPIST_Stellar_inputs,MC_inputs_ar)

            except:
                print ('didnt work ',i)
                #pdb.set_trace()
                outs = []

    print ('done with ',i)
    return outs

Everything = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs) #Run paralllized code
input_mega=[] # Collect input parameters for saving

for kj in range(0,len(inputs)):
    print ('saving garbage',kj)
    load_name = 'switch_garbage3/inputs4L%d.npy' %kj
    input_mega.append(np.load(load_name,allow_pickle=True))


## Save outputs for processing
np.save('MC_outputs_Trappist',Everything) 
np.save('MC_inputs_Trappist',input_mega)  

print ("All model runs completed!")

shutil.rmtree('switch_garbage3')



