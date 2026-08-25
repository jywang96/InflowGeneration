import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt
    
rDictInflow = {'52':'21p55'
              ,'57':'23p05'
              ,'62':'24p55'
              ,'67':'26p05'
              ,'72':'27p55'
              ,'77':'29p05'
              ,'82':'30p55'
              ,'87':'32p05'
              ,'92':'33p55'}

rDictALF    = {'52':'23p65'
              ,'57':'25p15'
              ,'62':'26p65'
              ,'67':'28p15'
              ,'72':'29p65'
              ,'77':'31p15'
              ,'82':'32p65'
              ,'87':'34p15'
              ,'92':'35p65'}

yMax = 1.5

for h in [0.04,0.06,0.08,0.12,0.14,0.16]:
    
    if h in [0.04,0.08,0.12,0.16]:
        rList = [52,62,72,82,92]
    
    elif h in [0.06,0.14]:
        rList = [57,67,77,87]
        
    my_dpi = 100
    plt.figure(figsize=(2260/my_dpi, 1300/my_dpi), dpi=my_dpi)
    for r in rList:
        
        all_df = pd.DataFrame()
        #inflow_df = pd.DataFrame()
        
        for pfx in [rDictALF[str(r)], rDictInflow[str(r)]]:
            
            temp = pd.DataFrame()

            directory = '../../PFCoarseABL/PFh'+'{0:.2f}'.format(h)+'/'

            temp['x-velocity']           = np.loadtxt(directory+pfx+'_avg_u.00100000.collapse_width.dat',skiprows = 3)[:,5]
            temp['x-velocity-magnitude'] = np.loadtxt(directory+pfx+'_mag_u.00100000.collapse_width.dat',skiprows = 3)[:,5]
            temp['uu-reynolds-stress']   = np.loadtxt(directory+pfx+'_rms_u.00100000.collapse_width.dat',skiprows = 3)[:,5]**2
            temp['vv-reynolds-stress']   = np.loadtxt(directory+pfx+'_rms_v.00100000.collapse_width.dat',skiprows = 3)[:,5]**2
            temp['ww-reynolds-stress']   = np.loadtxt(directory+pfx+'_rms_w.00100000.collapse_width.dat',skiprows = 3)[:,5]**2
            temp['uv-reynolds-stress']   = np.loadtxt(directory+pfx+'_uv.00100000.collapse_width.dat',   skiprows = 3)[:,5]
            
            #temp['x-velocity'] = avg_u
            temp['pseudo_Iu']  = np.sqrt(temp['uu-reynolds-stress'])/temp['x-velocity']
            temp['pseudo_Iv']  = np.sqrt(temp['vv-reynolds-stress'])/temp['x-velocity']
            temp['pseudo_Iw']  = np.sqrt(temp['ww-reynolds-stress'])/temp['x-velocity']
            temp['pseudo_Iuv'] = np.sqrt(np.abs(temp['uv-reynolds-stress']))/temp['x-velocity']
            
            # The two pseudo-coordinates place the upstream planes in the
            # DOWNSTREAM domain's frame, whose origin is the end of the
            # roughness canopy at upstreamLength = lBox + 0.5*spacing + lFetch
            # = 2.1 + 0.15 + 2.7 = 4.95 m.  So the downstream inlet (domain
            # x = 0) is at -4.95 and the end of the ALF box (domain x = 2.1)
            # is at -2.85.  rDictInflow is the upstream station that feeds the
            # inlet and is therefore -4.95; rDictALF sits 2.1 m (7 rows)
            # further downstream and is -2.85.
            #
            # These two assignments were swapped, which mislabelled the inlet
            # profile as the ALF target and vice versa.  Three independent
            # checks fix the orientation:
            #   * generateInflow.py writes x = -4.95 to *_inflow_input.txt and
            #     x = -2.85 to *_ALF_input.txt;
            #   * the superseded path at the bottom of this file assigned
            #     inflow_df -> -4.95 and ALF_df -> -2.85;
            #   * in the committed GPRDatabase, x = -2.85 carries a larger
            #     momentum deficit and more turbulence aloft than x = -4.95,
            #     i.e. it is the further-downstream station.
            if pfx == rDictInflow[str(r)]:
                temp['x'] = -4.95
            elif pfx == rDictALF[str(r)]:
                temp['x'] = -2.85
            else:
                raise ValueError(f'prefix {pfx} matches neither station for r={r}')

            temp['y'] = np.loadtxt(directory+pfx+'_avg_u.00100000.collapse_width.dat',skiprows = 3)[:,3]
            temp['z'] = 0
            temp['h'] = h
            temp['r'] = r
            # True row index in the upstream canopy.  The label r names the ALF
            # station; the inlet station sits 7 rows (2.1 m) upstream of it.
            temp['n'] = r if pfx == rDictALF[str(r)] else r - 7
            temp['y-velocity'] = 0
            temp['z-velocity'] = 0
            temp['uw-reynolds-stress'] = 0
            temp['vw-reynolds-stress'] = 0
            temp = temp.sort_values(by='y', ascending=True)
            all_df = pd.concat([temp, all_df], ignore_index=True)
            
        for x in [0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,4.0,5.0,6.0,7.0,9.0,11.0,13.0]:
            
            temp = pd.DataFrame()
        
            pfx = f"{x:.1f}".replace('.', 'p')
            
            directory ='../../TIGTestMatrixLong/PFh'+'{0:.2f}'.format(h)+'u15r'+'{0:.0f}'.format(r)+'/'

            temp['x-velocity']           = np.loadtxt(directory+pfx+'_avg_u.00075000.collapse_width.dat',skiprows = 3)[:,5]
            temp['x-velocity-magnitude'] = np.loadtxt(directory+pfx+'_mag_u.00075000.collapse_width.dat',skiprows = 3)[:,5]
            temp['uu-reynolds-stress']   = np.loadtxt(directory+pfx+'_rms_u.00075000.collapse_width.dat',skiprows = 3)[:,5]**2
            temp['vv-reynolds-stress']   = np.loadtxt(directory+pfx+'_rms_v.00075000.collapse_width.dat',skiprows = 3)[:,5]**2
            temp['ww-reynolds-stress']   = np.loadtxt(directory+pfx+'_rms_w.00075000.collapse_width.dat',skiprows = 3)[:,5]**2
            temp['uv-reynolds-stress']   = np.loadtxt(directory+pfx+'_uv.00075000.collapse_width.dat',   skiprows = 3)[:,5]
            
            #temp['x-velocity'] = avg_u
            temp['pseudo_Iu']  = np.sqrt(temp['uu-reynolds-stress'])/temp['x-velocity']
            temp['pseudo_Iv']  = np.sqrt(temp['vv-reynolds-stress'])/temp['x-velocity']
            temp['pseudo_Iw']  = np.sqrt(temp['ww-reynolds-stress'])/temp['x-velocity']
            temp['pseudo_Iuv'] = np.sqrt(np.abs(temp['uv-reynolds-stress']))/temp['x-velocity']
            
            temp['x'] = x
                
            temp['y'] = np.loadtxt(directory+pfx+'_avg_u.00075000.collapse_width.dat',skiprows = 3)[:,3]
            temp['z'] = 0
            temp['h'] = h
            temp['r'] = r
            # Downstream stations have no upstream row index.
            temp['n'] = np.nan
            temp['y-velocity'] = 0
            temp['z-velocity'] = 0
            temp['uw-reynolds-stress'] = 0
            temp['vw-reynolds-stress'] = 0
            temp = temp.sort_values(by='y', ascending=True)
            all_df = pd.concat([temp, all_df], ignore_index=True)
            
        all_df.to_csv('GPRDatabaseABL/'+'{0:.2f}'.format(h)+'u15r'+'{0:.0f}'.format(r)+'.csv'
                    , columns=['x','y','z','x-velocity','y-velocity','z-velocity','x-velocity-magnitude'
                              ,'uu-reynolds-stress','vv-reynolds-stress','ww-reynolds-stress'
                              ,'uv-reynolds-stress','uw-reynolds-stress','vw-reynolds-stress'
                              ,'pseudo_Iu','pseudo_Iv','pseudo_Iw','pseudo_Iuv','h','r','n'], index=False)
        
        #inflow_df = (pd.read_csv('../../TIGTestMatrixLong/InflowProfiles/Dragh'+'{0:.2f}'.format(h)+'_'+rDictInflow[str(r)]+'_inflow_turbulence.txt',sep='\t'))
        #inflow_df = (pd.read_csv('../../TIGTestMatrixLong/InflowProfiles/Dragh'+'{0:.2f}'.format(h)+'_'+rDictInflow[str(r)]+'_inflow_turbulence.txt',sep='\t'))
        #inflow_df['uv-reynolds-stress'] = np.abs(inflow_df['uv-reynolds-stress'])
        #inflow_df['pseudo_Iu']  = np.sqrt(inflow_df['uu-reynolds-stress'])/inflow_df['x-velocity']
        #inflow_df['pseudo_Iv']  = np.sqrt(inflow_df['vv-reynolds-stress'])/inflow_df['x-velocity']
        #inflow_df['pseudo_Iw']  = np.sqrt(inflow_df['ww-reynolds-stress'])/inflow_df['x-velocity']
        #inflow_df['pseudo_Iuv'] = np.sqrt(inflow_df['uv-reynolds-stress'])/inflow_df['x-velocity']
        #inflow_df['x-velocity-magnitude'] = -1
        #inflow_df['x'] = -4.95
        #inflow_df['h'] = h
        #inflow_df['r'] = r
        
        ##ALF_df    = (pd.read_csv('../../TIGTestMatrixLong/InflowProfiles/Dragh'+'{0:.2f}'.format(h)+'_'+rDictALF[str(r)]+'_inflow_turbulence.txt',sep='\t'))
        #ALF_df    = (pd.read_csv('../../TIGTestMatrixLong/InflowProfiles/Dragh'+'{0:.2f}'.format(h)+'_'+rDictALF[str(r)]+'_inflow_turbulence.txt',sep='\t'))
        #ALF_df['uv-reynolds-stress'] = np.abs(ALF_df['uv-reynolds-stress'])
        #ALF_df['pseudo_Iu']  = np.sqrt(ALF_df['uu-reynolds-stress'])/ALF_df['x-velocity']
        #ALF_df['pseudo_Iv']  = np.sqrt(ALF_df['vv-reynolds-stress'])/ALF_df['x-velocity']
        #ALF_df['pseudo_Iw']  = np.sqrt(ALF_df['ww-reynolds-stress'])/ALF_df['x-velocity']
        #ALF_df['pseudo_Iuv'] = np.sqrt(ALF_df['uv-reynolds-stress'])/ALF_df['x-velocity']
        #ALF_df['x-velocity-magnitude'] = -1
        #ALF_df['x'] = -2.85
        #ALF_df['h'] = h
        #ALF_df['r'] = r
        
        ##database_df = (pd.read_csv('../GPRDatabase/PFh'+'{0:.2f}'.format(h)+'u15r'+'{0:.0f}'.format(r)+'.txt',sep='\t'))
        #database_df = (pd.read_csv('../GPRDatabase/PFh'+'{0:.2f}'.format(h)+'u15r'+'{0:.0f}'.format(r)+'.txt',sep='\t'))
        #database_df['pseudo_Iu']  = np.sqrt(database_df['uu-reynolds-stress'])/database_df['x-velocity']
        #database_df['pseudo_Iv']  = np.sqrt(database_df['vv-reynolds-stress'])/database_df['x-velocity']
        #database_df['pseudo_Iw']  = np.sqrt(database_df['ww-reynolds-stress'])/database_df['x-velocity']
        #database_df['pseudo_Iuv'] = np.sqrt(database_df['uv-reynolds-stress'])/database_df['x-velocity']
        
        #allDFs = pd.concat([inflow_df, ALF_df, database_df], ignore_index=True)
        
        ##allDFs.to_csv('./GPRDatabase/PFh'+'{0:.2f}'.format(h)+'u15r'+'{0:.0f}'.format(r)+'.txt', sep='\t', index=False)
        
    
        #cont = 231
        #for QoI in ['x-velocity-magnitude','x-velocity','uu-reynolds-stress','vv-reynolds-stress','ww-reynolds-stress','uv-reynolds-stress']:
            #plt.subplot(cont)
            ##plt.plot(inflow_df[QoI],inflow_df['y'], label = 'Inflow'+rDictInflow[str(r)])
            ##plt.plot(ALF_df[QoI],ALF_df['y'], label = 'A'+rDictALF[str(r)])
            #x = -4.95
            #yPlot = np.sqrt(allDFs.loc[(abs(allDFs['x'] - x) < 1e-6) & (allDFs['y']<yMax), ['y']])
            ##plt.plot(database_df.loc[(abs(database_df['x'] - x) < 1e-6),[QoI]]/database_df.loc[(abs(database_df['x'] - x) < 1e-6),['x-velocity-magnitude']].to_numpy(),database_df.loc[(abs(database_df['x'] - x) < 1e-6), ['y']], label = 'uMag, '+str(r))
            ##plt.plot(database_df.loc[(abs(database_df['x'] - x) < 1e-6),[QoI]]/database_df.loc[(abs(database_df['x'] - x) < 1e-6),['x-velocity']].to_numpy(),database_df.loc[(abs(database_df['x'] - x) < 1e-6), ['y']], label = 'ux, '+str(r))
            #plt.plot(allDFs.loc[(abs(allDFs['x'] - x) < 1e-6) & (allDFs['y']<yMax),[QoI]],yPlot, label = 'ux, '+str(r))
            #plt.title(QoI)
            #cont +=1
    #plt.suptitle('h = ' +str(h))
    #plt.legend()
        
    #plt.show()
    #plt.close('all')
        
        
        