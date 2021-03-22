#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:04:14 2020
The program interpolate geolocation variables from new metop proxy data 
from reduced spatial resolution to the full resolution. 
the process flow is 
1. convert lat&lon from geodetic to cartesian XYZ.
2. bilinear interpolate X, Y, Z into xfull, yfull, zfull
3. convert cartesian xfull, yfull, zfull back to geodetic lat& lon
4. interpolate other geo-variable solar zenith, solar azimuth, sensor zenith, 
sensor azimuth to full spatial resolution. 
Notes: bilinear interpolation is 1D array, generate a function,then using 
function to get new resolution data. 

Copy radiance data and metadata to *.fullrad.nc 
@author: yan.zhang
"""

from glob import glob
from netCDF4 import Dataset
from scipy import interpolate
import os
import time
import numpy as np
#for parallel running


#convert reduced lats/lons to full lats/lons
def METIMAGE_RECONSTRUCT_GEOLOCATION(gridALT, gridACT, latsReduced, lonsReduced):
######################################  
#convert lat lon to X,Y,Z
    deg2Rad=np.pi/180
    a=6378137.0
    b=6356752.3
    
    a2=a*a
    b2=b*b
    e=np.sqrt(1-b2/a2)
    e2=e*e
    
    latsReducedR=latsReduced*deg2Rad
    lonsReducedR=lonsReduced*deg2Rad
    sinlat=np.sin(latsReducedR)
    sinlat2=sinlat*sinlat
    coslat=np.cos(latsReducedR)
    sinlon=np.sin(lonsReducedR)
    coslon=np.cos(lonsReducedR)
    
    N=a/np.sqrt(1-e2*sinlat2)
#    N1=a/(1-e2*sinlat2)
    X=N*coslat*coslon
    Y=N*coslat*sinlon
    Z=(1-e2)*N*sinlat; #equestion 7 
#    print('X',np.shape(X),np.shape(Y))
#    print(latsReduced[10:19,0])
#    print(X[10:19,0],Y[10:19,0],Z[10:19,0])
    ##################################
    
    indx_xx=np.arange(394)
    indx_yy=np.arange(700)
#    indx_x=np.zeros((394, 700))
#    indx_y=np.zeros((394, 700))
    xFull = np.zeros([3144,4200])
    yFull = np.zeros([3144,4200])
    zFull = np.zeros([3144,4200])
#    for i in range(700):
#        indx_x[:,i]=indx_xx;
#    for j in range(394):
#        indx_y[j,:]=indx_yy;
#    xx, yy = np.meshgrid(indx_yy, indx_xx)
#    print('indx_x:', np.shape(xx), np.shape(yy),np.shape(Z))
    print('gridACT:', np.shape(gridACT))
    
#    print('gridACT: ', gridACT)
#    print('gridALT: ', gridALT)
    xFull_f=interpolate.interp2d(indx_yy, indx_xx, X)
    yFull_f=interpolate.interp2d(indx_yy, indx_xx, Y)
    zFull_f=interpolate.interp2d(indx_yy, indx_xx, Z)
    xFull = xFull_f(gridALT[0,:],gridACT[:,0])
    yFull = yFull_f(gridALT[0,:],gridACT[:,0])
    zFull = zFull_f(gridALT[0,:],gridACT[:,0])
#    xFull = xFull_f(gridACT[:,0],gridALT[0,:])
#    yFull = yFull_f(gridACT[:,0],gridALT[0,:])
#    zFull = zFull_f(gridACT[:,0],gridALT[0,:])
    ###################################

    xFull2=xFull*xFull
    yFull2=yFull*yFull
    p=np.sqrt(xFull2+yFull2)
    tmp1=yFull/(xFull+p)
    lonsFullR=2*np.arctan(tmp1)
    lonsFull=lonsFullR/deg2Rad
    
    e1=np.sqrt(a2/b2-1) ; #equation 9-10
    e12=e1*e1
    theta=np.arctan((zFull*a)/(p*b))
    sintheta=np.sin(theta)
    sintheta3=sintheta*sintheta*sintheta
    costheta=np.cos(theta)
    costheta3=costheta*costheta*costheta
#    acostheta=np.acos(theta)
    latsFullR=np.arctan((zFull+e12*b*sintheta3)/(p-e2*a*costheta3))
    latsFull=latsFullR/deg2Rad

    return latsFull, lonsFull
  
######################################
# calculate tie point grid indices for full geolocation
def METIMAGE_CALCULATE_RECONSTRUCT_GEOLOCATION_GRIDS(Zalt1, Zact1):
    #MetImage constants
    Zalt=Zalt1  #pixel per zone in the scan direction
    Zact= Zact1 #pixel per zone in the track direction
    DIMact=3144  # of pixels in scan direction
    DIMalt=4200  ;# of pixels in track direction
    nDetectors=24 ;#of detectors

#    nzone_act=DIMact/Zact
#    nzone_alt=DIMalt/Zalt
    nScans=int(DIMalt/nDetectors)
    nTiePointsPerScanAlt=int(nDetectors/Zalt+1)

#calculate float index value to bilinear interpolation
    gridALT=np.zeros((DIMact, DIMalt))
    gridACT=np.zeros((DIMact, DIMalt))
#    gridALT=[[0 for c in range(DIMalt-1)] for r in range(DIMact-1)]
#    gridACT=[[0 for c in range(DIMalt-1)] for r in range(DIMact-1)]
    #alt direction
    array=np.zeros(DIMalt)
    
    fy=np.arange(Zalt)*1.0/Zalt
    zone=0
    index1=0
    for iscan in range(nScans):
        for j in range(nTiePointsPerScanAlt-1):
            index2=index1+Zalt
#            print('index1:index2 :',index1, index2)
#            print(fy,'array = ', array[index1:index2])
            array[index1:index2]=fy+zone
            zone=zone+1
            index1=index2
        zone=zone+1
    
    for i in range(DIMact):
        gridALT[i,:]=array
#    print('gridALT: ', gridALT)
    
    #act direction
    fx=np.arange(DIMact)*1.0/Zact
    print('fx',np.shape(fx),np.shape(gridACT[0,:]))
    
    for i in range(DIMalt):
        gridACT[:,i]=fx
  
    return gridALT,gridACT


######################################
def FULL_GEO_CONVERT(Zalt1, Zact1,latsReduced,lonsReduced,lat_scale_factor,lon_scale_factor):
        gridALT, gridACT = METIMAGE_CALCULATE_RECONSTRUCT_GEOLOCATION_GRIDS(Zalt1, Zact1)
        #interpolate to full resolutions
        latsReduced =latsReduced*lat_scale_factor
        lonsReduced =lonsReduced*lon_scale_factor
        latsFull, lonsFull = METIMAGE_RECONSTRUCT_GEOLOCATION(gridALT, gridACT, latsReduced, lonsReduced)
        return latsFull, lonsFull
        
def ReadFileNames(path, wildchar):
    return  [y for x in os.walk(path) for y in glob(os.path.join(x[0], wildchar))]

####################################################
# METIMAGE_L1B_READER_MAIN
   
dirIn='/data/smcd7/yzhang/METOP_SG_Proxy_Data/SimulatedMetImage_2/'
dirOut_geo=dirIn+'converted/geo/'
if not os.path.exists(dirOut_geo):
    os.mkdir(dirOut_geo,0o755)
dirOut_rad=dirIn+'converted/rad/'
if not os.path.exists(dirOut_rad):
    os.mkdir(dirOut_rad,0o755)
    
bandNamesVII=['vii_443','vii_555','vii_668','vii_752','vii_763',\
              'vii_865','vii_914','vii_1240','vii_1375','vii_1630',\
              'vii_2250','vii_3740','vii_3959','vii_4050','vii_6725',\
              'vii_7325','vii_8540','vii_10690','vii_12020','vii_13345']
bandNamesVII_rename=['metim_00443','metim_00555','metim_00668','metim_00752','metim_00763',\
              'metim_00865','metim_00914','metim_01240','metim_01375','metim_01630',\
              'metim_02250','metim_03740','metim_03959','metim_04050','metim_06725',\
              'metim_07325','metim_08540','metim_10690','metim_12020','metim_13345']
varNamesGEO=['solar_zenith','solar_azimuth',\
             'observation_zenith','observation_azimuth']
varNamesGEO_1=['latitude','longitude']
nBands=int(len(bandNamesVII)); 
    
files_in_dir = []
cnt = 0

       
files_in_dir = ReadFileNames(dirIn,'W_xx-noaa-star*T_N____.nc')
#files_in_dir = ReadFileNames(dirIn,'W_xx-noaa-star*G_D_20180915121500*T_N____.nc')
n_file = int(len(files_in_dir));
print('n_file = ', n_file)

Zalt1 = 8;
Zact1 = 8;
fin_grps = 'data/measurement_data/';
fout_grps= "VII_SWATH_Type_L1B/Data_Fields/";
fout_grps_geo= "VII_SWATH_Type_L1B/Geolocation_Fields/";

startime = time.time()
for i_file in range(0,1):
#for i_file in range(0,n_file):
    if Zalt1 == 8:
        filenameIn = files_in_dir[i_file]
        loc_1 = filenameIn.rfind('/')
        pathnameOut_rad=dirOut_rad+filenameIn[loc_1:-3]+'.fullrad.nc'
    
        #read reduced resolution geolocation (in tie point representation)
        fin = Dataset(filenameIn)
         
        lat_scale_factor = 3.433241E-4
        lon_scale_factor = 3.4332342E-4
        sza_scale_factor = 0.0054934993
        latsReduced = fin[fin_grps].variables['latitude'][:]
        lonsReduced = fin[fin_grps].variables['longitude'][:]

        latsReduced =latsReduced*lat_scale_factor
        lonsReduced =lonsReduced*lon_scale_factor
        gridALT, gridACT = METIMAGE_CALCULATE_RECONSTRUCT_GEOLOCATION_GRIDS(Zalt1, Zact1)
        latsFull, lonsFull = METIMAGE_RECONSTRUCT_GEOLOCATION(gridALT, gridACT, latsReduced, lonsReduced)
        latsFull = latsFull/lat_scale_factor
        lonsFull = lonsFull/lon_scale_factor
        
        n_row, n_col = np.shape(latsReduced)
        indx_row = np.arange(n_row)
        indx_col = np.arange(n_col)
        
        f_rad = Dataset(pathnameOut_rad,"w", format="NETCDF4")
        f_rad = Dataset('test.nc',"w", format="NETCDF4")
        
        dim1,dim2=np.shape(latsFull)
        lat2 = f_rad.createDimension('lat', dim1) # Unlimited
        lon2 = f_rad.createDimension('lon', dim2) # Unlimited

#        NLat = f_geo.createVariable(fout_grps_geo+"Latitude",latsReduced.datatype,("lat","lon",))
#        NLon = f_geo.createVariable(fout+grps_geo+"Longitude",lonsReduced.datatype,("lat","lon",))
#        NLat[::] = latsFull[::]
#        NLon[::] = lonsFull[::]
        
            
        for name, variable in fin[fin_grps].variables.items():
            if name in bandNamesVII:
                loc_find = name.find('vii_')
                numband  =  int(name[loc_find+4::])
                if numband == 668:
                    numband = 670;
        
                strband  = str(numband).zfill(5);
                name_new = 'metim_'+strband;            
                if '_FillValue' in fin[fin_grps+name].ncattrs():
                    fill_value = fin[fin_grps+name]._FillValue
                    x = f_rad.createVariable(fout_grps+name_new, variable.datatype, ("lat","lon",), fill_value=fill_value)
#                    print("   Variable created with _FillValue attr:", fill_value)
                else:
                    x = f_rad.createVariable(fout_grps+name_new, variable.datatype, ("lat","lon",))         

                x[::]=fin[fin_grps].variables[name][::]

                for attr in fin[fin_grps+name].ncattrs():
                    if '_FillValue' in attr:
                        continue
#                    print("   Adding attribute:", attr, '=', fin['data/measurement_data/'+name].getncattr(attr))
                    f_rad['VII_SWATH_Type_L1B/Data_Fields/'+name_new].setncattr(attr, fin['data/measurement_data/'+name].getncattr(attr))
        f_rad.close()
        print('Dataset2 is closes')                    

        endtime = time.time()
