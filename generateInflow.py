# Script to generate inflow profiles using Gaussian Process Regression (GPR)
# and prepare corresponding case geometry and boundary‐condition files.

# ===== Imports & Library Setup =====
import os
import sys
import joblib
# Import data processing and plotting libraries
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# Import application‐specific GPR and mesh modules
from modelDefinition import *
from stl import mesh
from hyperparametersGPR import features, xList, hTrain, rTrain, devPairs, testPairs, trainPairs, uncertainty, intensitiesModelID, inflowModelID
import json

# Path to the precomputed GPR feature database
PFDatabase = './GPRDatabase'

# ===== Reference Case Configuration =====
###### This is for reference, now loaded from caseConfig.json ######
# Reference case setups (multiple options commented out; select one below)
#### themisABL setup ####
#reference = {'fName':'themisABL'
            #,'h':0.06,'r':87,'alpha':0.4,'k':1.5,'x':0.9,'hMatch':0.714}

#### themisABL_alpha setup ####
#reference = {'fName':'themisABL_alpha'
            #,'h':0.04,'r':92,'alpha':0.20,'k':1.5,'x':0.3,'hMatch':0.714}

##### TPU_highrise_14_middle_dimensional setup ####
#reference = {'fName':'TPU_highrise_14_middle_dimensional'
             #,'h':0.04,'r':92,'alpha':0.7,'k':1.11,'x':2.7,'hMatch':0.714}

##### LRB Cat1 scale 1:50 ####
#reference = {'fName':LRB_Cat1_scale1to50
             #,'h':0.12,'r':79,'alpha':0.18,'k':1.62,'x':11.0,'hMatch':0.4}

##### LRB Cat1 scale 1:25 ####
#reference = {'fName':'LRB_Cat1_scale1to25'
             #,'h':0.04,'r':75,'alpha':0.36,'k':1.36,'x':3.3,'hMatch':0.666}

###### LRB Cat1 scale 1:10 ####
##reference = {'fName':'LRB_Cat1_scale1to10'
             ##,'h':0.16,'r':64,'alpha':0.90,'k':1.25,'x':9.0,'hMatch':0.666}

###### MRB Cat2 scale 1:100 ####
##reference = {'fName':'MRB_Cat2_scale1to100'
             ##,'h':0.04,'r':92,'alpha':0.45,'k':1.39,'x':3.30,'hMatch':0.666}

###### MRB Cat3 scale 1:100 ####
##reference = {'fName':'MRB_Cat3_scale1to100'
             ##,'h':0.08,'r':92,'alpha':0.45,'k':1.56,'x':1.50,'hMatch':0.666}

##### LRB Cat1 ####
#fName = 'LRB_Cat1'
#reference = {'fName':'LRB_Cat1'
             #,'h':0.06,'r':54,'alpha':0.42,'k':1.35,'x':3.3,'hMatch':0.666}

##### LRB Cat2 ####
#fName = 'LRB_Cat2'
#reference = {'fName':'LRB_Cat2'
             #,'h':0.04,'r':89,'alpha':0.23,'k':1.75,'x':0.6,'hMatch':0.666}

##### MRB Cat2 ####
#fName = 'MRB_Cat2'
#reference = {'fName':'MRB_Cat2'
             #,'h':0.04,'r':92,'alpha':0.42,'k':1.47,'x':3.0,'hMatch':0.666}


##### HRB Cat2 ####
#fName = 'HRB_Cat4'
#reference = {'fName':'HRB_Cat4'
             #,'h':0.05,'r':52,'alpha':0.25,'k':1.52,'x':0.6,'hMatch':0.666}

#### Frank Cat_4 ####
# fName = 'BL-1_0_alpha'
# reference = {'fName':'BL-1_0_alpha'
#              ,'h':0.13,'r':91,'alpha':0.61,'k':1.48,'x':4.0,'hMatch':0.666}

##### MRB CatB ####
# fName = 'MRB_Cat_B'
# reference = {'fName':fName
#             ,'h':0.04,'r':92,'alpha':0.42,'k':1.47,'x':3.0,'hMatch':0.666}

caseConfig = json.load(open('caseConfig.json'))
reference = caseConfig['reference']
scaleFactors = caseConfig['scaleFactors']
plotABL = caseConfig.get('plotABL', False)
wDomain = caseConfig.get('wDomain', 3.0)
# ===== Scaling Factor Computation =====
# Compute geometric scaling factors from HABL and reference α
scale = scaleFactors['scale']
#scale = 1.0/21.42857142857111
#HABL = 240.0
H_build = scaleFactors['H_build']
HABL = H_build * 1.5

# --- velocity scale -------------------------------------------------------
# The surrogate returns dimensionless profiles, u / U_inf.  Writing the inlet
# boundary condition as q(y) * V makes every velocity in the simulation
# proportional to V, so the ABL at the building is p_U(eta) * V.  Requiring
# that it equal the target at the reference height,
#
#     p_U(eta_ref) * V = U_target(y_ref)      =>      V = s_U * U_target(y_T)
#
# with s_U = t_U(eta_ref) / p_U(eta_ref) the dimensionless scale the optimizer
# reports.  U_target(y_T) is a property of the target profile, not a free
# parameter, which is why the former 'Uscaling' entry has been removed from
# caseConfig.json: it was a user-set constant standing in for a quantity the
# target file already determines.
#
# 'y_ref' is given in the units of the target's y column and defaults to the
# building height.  Do not ask the user for the ratio y_ref / y_T: the same
# building gives a different ratio depending on how far above it the target
# profile was specified.
y_ref = reference.get('y_ref', H_build)

scaling = HABL*scale/reference['alpha']
caseDirectory = './'+reference['fName']+'_geometric_1to'+str(np.round(1.0/scale).astype(int))
# Save the reference dictionary as a JSON file in the case directory

generateCase(scaling, reference['h'], reference['x'], caseDirectory, reference['fName'], wDomain=wDomain)
with open(f"{caseDirectory}/caseConfig.json", 'w') as json_file:
    json.dump(caseConfig, json_file, indent=4)

# Set maximum non‐dimensional y for normalization in GPR
yMax= 1.0

# Prepare DataFrame of features at the reference case for prediction
fit_features = pd.DataFrame()
fit_features['y'] = np.linspace(0.01,1.0,2000)
fit_features['x'] = reference['x']
fit_features['h'] = reference['h']
fit_features['alpha'] = reference['alpha']
fit_features['r'] = reference['r']
fit_features['k'] = reference['k']

# ===== Assemble Training, Development, and Test Datasets =====
# Build parameter pairs and pack into dicts for GPR
# Assemble dictionaries of training, development, and test points
devPoints = {'h':devPairs[:,0],'r':devPairs[:,1],'x':[reference['x']]}
testPoints = {'h':testPairs[:,0],'r':testPairs[:,1],'x':[reference['x']]}
trainPoints = {'h':trainPairs[:,0],'r':trainPairs[:,1],'x':[reference['x']]}

# Initialize the GaussianProcess object with our data sets
gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase)

# ===== Plot ABL Profiles (if enabled) =====
my_dpi = 100
def plotABL(reference, save=False):
    # -- Load & normalize reference ABL data (plot only) --
    prefix   = str(reference['x']).replace('.', 'p') + '_'
    directory = prefix + intensitiesModelID
    ref_abl  = pd.read_csv('TestCases/'+reference['fName']+'.dat', sep=',')
    header   = list(ref_abl.columns)
    idx      = np.argmax(ref_abl['y'].to_numpy())
    yref     = ref_abl['y'].iloc[idx]
    ref_abl['y'] = ref_abl['y'] / yref

    # -- Check the anchoring: after scaling by s_U the candidate and the
    #    target must agree at y_ref, by construction of the optimizer.
    eta_ref  = y_ref / yref
    model    = '../GPRModels/'+directory+'_u.pkl'
    y_mean   = gp.predict(model, fit_features, features, 'u')
    y_mean   = y_mean.loc[y_mean['y'] <= reference['alpha'] * y_mean['y'].max()]
    y_mean['y'] = y_mean['y'] / y_mean['y'].max()

    U_ABL_nd = interp1d(ref_abl['y'], ref_abl['u'] / ref_abl['u'].iloc[idx])(eta_ref).item()
    U_TIG_nd = interp1d(y_mean['y'],  y_mean['y_model'])(eta_ref).item()
    mismatch = reference['k'] * U_TIG_nd / U_ABL_nd - 1.0

    print(f'anchor: y_ref = {y_ref:.3f} m  (eta_ref = {eta_ref:.4f})')
    print(f'        target {U_ABL_nd:.4f}  vs  s_U * candidate '
          f'{reference["k"] * U_TIG_nd:.4f}   mismatch {100*mismatch:+.3f}%')
    if abs(mismatch) > 5e-3:
        print('        WARNING: s_U in caseConfig.json does not anchor this '
              'target at y_ref; re-run the optimizer or check y_ref.')

    # -- Configure figure and plot profiles --
    plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)
    cont = 1
    for QoI in ['u','Iu','Iv','Iw']:
        # Predict each quantity of interest (QoI) with GPR
        model = '../GPRModels/'+directory+'_'+QoI+'.pkl'
        
        y_mean = gp.predict(model,fit_features,features,QoI)
        H_norm = reference['alpha']*np.max(y_mean['y'])
        print(f"For ABL plot, y normalized by H = {H_norm:.3f} m, or {scaling*H_norm:.3f} m in simulation space")
        y_mean = y_mean.loc[y_mean['y']<=H_norm]
        y_mean['y'] = y_mean['y']/(y_mean['y'].max())
        
        # Apply appropriate scaling to the predicted profiles
        if QoI == 'u':
            y_mean['y_model'] = y_mean['y_model']*reference['k']
            y_mean['y_std'] = y_mean['y_std']*reference['k']
            print(f"For ABL plot, U scaled by s_U = {reference['k']:.3f}")
            
        plt.subplot(1,4,cont)
        
        if (QoI in header):
            
            if QoI == 'u':
                plt.plot(ref_abl[QoI]/ref_abl['u'].iloc[idx],ref_abl['y'],color='tab:red',label='Target',linewidth=3)
                plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]/ref_abl['u'].iloc[idx]*0.9, ref_abl[QoI]/ref_abl['u'].iloc[idx]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
            else:
                plt.plot(ref_abl[QoI],ref_abl['y'],color='tab:red',label='Target',linewidth=3)
                plt.fill_betweenx(ref_abl['y'], ref_abl[QoI]*0.9, ref_abl[QoI]*1.1, color='tab:red', alpha=0.2,label=r'Reference $\pm$10%')
            
        line = plt.plot(y_mean['y_model'],y_mean['y'],linestyle='--',linewidth=3
                ,label=r'x='+'{0:.2f}'.format(reference['x'])+'m,h='+'{0:.2f}'.format(reference['h'])+'m'+r'm,$\alpha$='+'{0:.2f}'.format(reference['alpha'])+r',r='+'{0:.2f}'.format(reference['r']))
        
        if QoI in header:
            max_x = np.ceil((1.2*max([np.max(ref_abl[QoI]),np.max(y_mean['y_model'])])*10000).astype(int))/10000
        else:
            max_x = np.ceil(1.2*np.max(y_mean['y_model'])*10000).astype(int)/10000
            
        if uncertainty == True:
            plt.fill_betweenx(y_mean['y'], y_mean['y_model']-2*y_mean['y_std'], y_mean['y_model']+2*y_mean['y_std'], color=line[0].get_color(), alpha=0.2)
        
        
        #plt.xlim(0,1.1*max_x)
        plt.xlabel(QoI)
        
        plt.ylim(0,1)
        plt.yticks([0.25,0.5,0.75,1.0])
        
        if QoI=='u'or QoI == 'Iv':
            plt.ylabel('y/H')
        else:
            plt.gca().set_yticklabels([])

        cont += 1

    plt.suptitle('Chosen setup, dimension vs adimensional y')

    plt.legend(frameon=False)
    if save:    
        plt.savefig('TestCases/'+reference['fName']+'.png', bbox_inches='tight')
    return

if plotABL:
    plotABL(reference, save=True)

# ===== Generate Geometric Case Files & Inflow Profiles =====
yMax = 1.5 # [m] Height of the GPR downstream fit (not yMax in the paper)
# redefine yMax to extend the vertical normalization range for inflow generation
        
# ===== Inflow Profile Generation & Export =====
# Anchor the velocity scale on the target at y_ref, then convert to physical
# units.  Both steps use the same number, so the profiles are multiplied once.
ref_abl_dim = pd.read_csv('TestCases/' + reference['fName'] + '.dat', sep=',')
_idx  = int(np.argmax(ref_abl_dim['y'].to_numpy()))
y_T   = float(ref_abl_dim['y'].iloc[_idx])
U_yT  = float(ref_abl_dim['u'].iloc[_idx])
U_ref = float(interp1d(ref_abl_dim['y'], ref_abl_dim['u'])(y_ref))

Vinlet = reference['k'] * U_yT
print(f'target: y_T = {y_T:.3f}, U(y_T) = {U_yT:.3f} m/s, '
      f'y_ref = {y_ref:.3f}, U(y_ref) = {U_ref:.3f} m/s')
print(f'velocity multiplier applied to the boundary conditions: '
      f'{Vinlet:.3f} m/s   (= s_U * U(y_T), s_U = {reference["k"]:.3f})')

plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)
for x in [-4.95, -2.85]:
    # Reinitialize GPR for a new inflow plane at x
    trainPoints = {'h': trainPairs[:,0], 'r': trainPairs[:,1], 'x': [x]}
    devPoints   = {'h': devPairs[:,0],   'r': devPairs[:,1],   'x': [x]}
    testPoints  = {'h': testPairs[:,0],  'r': testPairs[:,1],  'x': [x]}
    # redefine train/dev/test points dictionaries for the current x-location

    gp = gaussianProcess(trainPoints, devPoints, testPoints, yMax, PFDatabase, np.linspace(0.01,1.0,100))
    # reinitialize the GaussianProcess object for each inflow plane

    outputDF = pd.DataFrame()  
    fit_features = pd.DataFrame()
    # reset DataFrames inside the loop to accumulate new predictions per x

    fit_features['y'] = np.linspace(0.01/yMax, 1.0, 1501)
    fit_features['x'] = x
    fit_features['h'] = reference['h']
    fit_features['r'] = reference['r']
        
    outputDF['x'] = np.ones((len(fit_features['y'].to_numpy()),))*x
    outputDF['y'] = fit_features['y'].to_numpy()*yMax*scaling
    #outputDF['y'] = fit_features['y'].to_numpy()*yMax*reference['alpha']/yref
    outputDF['z'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['y-velocity'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['z-velocity'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['uw-reynolds-stress'] = np.zeros((len(fit_features['y'].to_numpy()),))
    outputDF['vw-reynolds-stress'] = np.zeros((len(fit_features['y'].to_numpy()),))

    cont = 1
    for QoI in ['u','uu','vv','ww','uv']:
        prefix = str(str(x)+'_').replace('.','p')
        directory = prefix + inflowModelID
        model = '../GPRModels/'+directory+'_'+QoI+'.pkl'  
        # redefine 'model' to load the QoI-specific GPR file each iteration

        y_mean = gp.predict(model, fit_features, features, QoI)
        y_mean['y'] = y_mean['y']/(y_mean['y'].max())
        
        if QoI == 'u':
            y_mean['y_model'] = y_mean['y_model']*Vinlet
            y_mean['y_std'] = y_mean['y_std']*Vinlet
        else:
            y_mean['y_model'] = y_mean['y_model']*Vinlet**2
            y_mean['y_std'] = y_mean['y_std']*Vinlet**2
            
        plt.subplot(2,3,cont)
        
        if x == -4.95:
            lab = 'inflow_input'
        elif x == -2.85:
            lab = 'ALF_input'
            
        if QoI == 'u':
            outputDF['x-velocity'] = y_mean['y_model'].to_numpy()
            line = plt.plot(outputDF['x-velocity'],outputDF['y'],linewidth=2,label=lab)
        elif QoI == 'uu':
            outputDF['uu-reynolds-stress'] = np.abs(y_mean['y_model'].to_numpy())
            line = plt.plot(outputDF['uu-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        elif QoI == 'vv':
            outputDF['vv-reynolds-stress'] = np.abs(y_mean['y_model'].to_numpy())
            line = plt.plot(outputDF['vv-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        elif QoI == 'ww':
            outputDF['ww-reynolds-stress'] = np.abs(y_mean['y_model'].to_numpy())
            line = plt.plot(outputDF['ww-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        elif QoI == 'uv':
            outputDF['uv-reynolds-stress'] = -np.abs(y_mean['y_model'].to_numpy())
            line = plt.plot(-outputDF['uv-reynolds-stress'],outputDF['y'],linewidth=2,label=lab)
        
        max_x = np.ceil(1.2*np.max(y_mean['y_model'])*10000).astype(int)/10000
        
        plt.xlim(0,1.1*max_x)
        
        if QoI=='u':
            plt.ylabel('y[m]')
            plt.xlabel('u[m/s]')
        else:
            plt.ylabel('y[m]')
            plt.xlabel(QoI+r'$[m^2/s^2]$')
        
        #if uncertainty == True:
            #plt.fill_betweenx(y_mean['y'], y_mean['y_model']-2*y_mean['y_std'], y_mean['y_model']+2*y_mean['y_std'], color=line[0].get_color(), alpha=0.2)
        
        cont +=1
    
    outputDF['y'] = outputDF['y']
    outputDF[['x','y','z','x-velocity','y-velocity','z-velocity','uu-reynolds-stress','vv-reynolds-stress','ww-reynolds-stress','uv-reynolds-stress','uw-reynolds-stress','vw-reynolds-stress']].to_csv(caseDirectory+'/'+reference['fName']+'_'+lab+'.txt',sep='\t',index=False)

plt.suptitle('Chosen setup')
plt.legend(frameon=False)
plt.show()
plt.close('all')