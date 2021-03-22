# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:50:23 2020

@author: Yan Zhang
"""

from pylab import *;
import os,time;
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
import numpy as np
from glob import glob
import matplotlib.pylab as plt

def bits_stripping(bit_start,bit_count,value):
        bitmask=pow(2,bit_start+bit_count)-1
        return np.right_shift(np.bitwise_and(value,bitmask),bit_start)

def resizeMono(xs, Mnew):
    from scipy import interpolate;
    N = len(xs);
    difxs = diff(xs);
    xstart = xs[0] - difxs[0]/2;
    xend = xs[-1] + difxs[-1]/2;
    difxsNewN = difxs*N/Mnew;
    xs_interp = interpolate.splrep(arange(N-1)/N*Mnew,difxsNewN,s=0);
    difxNewM = interpolate.splev(arange(Mnew-1), xs_interp, der =0);
    xsNew = xstart +cumsum(difxNewM);
    xsNew = array([xstart] + list(xsNew));
    return xsNew,difxNewM;
    
def ReadFileNames(path, wildchar):
    return  [y for x in os.walk(path) for y in glob(os.path.join(x[0], wildchar))]

def plot_cld_l2(lata, lona, vara, pstyle="k-"):
    yellow = np.array([11/12, 11/12, 5/12, 1])
    newcmp1 = ListedColormap(['b', yellow, 'g', 'lime'])
    varname = vara
    #set a figure
    fig=plt.figure(figsize=(13,6.2));
    fig.suptitle(varname, fontsize=18)
    ax = plt.subplot(121)
    mm = ax.pcolormesh( lona, lata,vara,vmin = 0, vmax = 400,cmap = newcmp1)
    cbar=fig.colorbar(mm, shrink=1.0, orientation='vertical',ticks=np.arange(0.5, 4, 1).tolist())
    cbar.ax.set_yticklabels(['Clouday', 'Pro cloudy', 'Pro clear', 'Clear'])
    #cbar=fig.colorbar(mm, shrink=1.1, orientation='vertical')   

def readHDF_geo(filename):
    hdf_geo = SD(filename, SDC.READ) 
    datasets_dic = hdf_geo.datasets()           
    sds_geo = hdf_geo.select('Latitude')
    lata = sds_geo.get()
    sds_geo = hdf_geo.select('Longitude')
    lona = sds_geo.get()
    return lata, lona;

def readHDF_data(filename, varname):
    hdf = SD(filename, SDC.READ) 
    datasets_dic = hdf.datasets() 
    sds_obj = hdf.select(varname)
    varname1 = sds_obj.get()
    return varname1;
    
def data_downsize(data, factor):
    """ downsize the large dataset by a factor
    for instance, data.shape = (21600, 43200)
    when factor =100, 
    it will downsize into a new data1.shape=(216, 432)
    """
    dim1, dim2 = data.shape
    dim1_new = int(dim1/factor);
    dim2_new = int(dim2/factor);

    data_new = np.empty((dim1_new, dim2_new));
    data_new[:] = np.nan
    Dim1s = np.arange(dim1_new);
    Dim2s = np.arange(dim2_new);

    Idx1 = Dim1s * int(dim1 / dim1_new);
    Idx1 = Idx1;
    for j in Dim2s:
        idx2 = j * dim2 / dim2_new
        data_new[Dim1s,j] = data[Idx1,int(idx2)]
    return data_new

class base_obj():
    def __init__(self):
        self.__data__ = {};
    
    def get(self,varname):
        value = None;
        if varname in self.__data__:
            value=self.__data__[varname];
        return value;
        
    def set(self,varname,value):
        self.__data__[varname] = value;
        return;
        
class SatGranule(base_obj):
    def __init__(self,latArray=None,lonArray=None,earthRadius=6378):
        base_obj.__init__(self);
        self.set('latArray',latArray);
        self.set('lonArray',lonArray);
        self.set('earthRadius',earthRadius);
        return;

    def lmdchis2xyz(self,lmds,chis):
        R = self.get('earthRadius');
        #lmds = lats*pi/180;
        #chis = lons*pi/180;
        xs = R*cos(lmds)*cos(chis);
        ys = R*cos(lmds)*sin(chis);
        zs = R*sin(lmds);
        return xs,ys,zs;
        
    def xyzs2lmdchi(self,xs,ys,zs):
        R = self.get('earthRadius');
        chis = arctan(ys/xs);
        if isscalar(chis):
            if xs<0:
                chis = chis + pi;
        else:
            I = xs<0;
            chis[I] =chis[I]+pi;
        lmds = arctan(zs/(xs**2+ys**2)**0.5);   
        return lmds,chis;
    
    def xyz2scanorbit(self,xs,ys,zs):
        mat = self.get('xyz2ScanorbitMatrix');
        center = self.get('scanorbitOrigin');
        #mat = inv(mat);
        xc, yc, zc = center;
        scns = mat[0,0]*(xs-xc)+mat[0,1]*(ys-yc)+mat[0,2]*(zs-zc);
        obts = mat[1,0]*(xs-xc)+mat[1,1]*(ys-yc)+mat[1,2]*(zs-zc);
        nrms = mat[2,0]*(xs-xc)+mat[2,1]*(ys-yc)+mat[2,2]*(zs-zc);
        return scns,obts,nrms; 
    
    def scanorbits2xyz(self,scns,obts,nrms):
        mat = self.get('xyz2ScanorbitMatrix');
        center = self.get('scanorbitOrigin');
        xc, yc, zc = center;
        xs = mat[0,0]*scns+mat[1,0]*obts+mat[2,0]*nrms+xc;
        ys = mat[0,1]*scns+mat[1,1]*obts+mat[2,1]*nrms+yc;
        zs = mat[0,2]*scns+mat[1,2]*obts+mat[2,2]*nrms+zc;
        return xs,ys,zs;    

    def lmdchis2scanorbit(self, lmds, chis):
        xs,ys,zs = self.lmdchis2xyz(lmds, chis);
        scns,obts,nrms = self.xyz2scanorbit(xs,ys,zs);
        return scns,obts,nrms;
    
    def scanorbit2lmdchis(self, scns,obts,nrms):
        xs,ys,zs = self.scanorbits2xyz(scns,obts,nrms);
        lmds,chis = self.xyzs2lmdchi(xs,ys,zs);
        return lmds,chis
    
    def plotXyz(self,xs,ys,zs,pstyle="-",grid=False):
        R = self.get('earthRadius');
        ax = self.get('globalAxes');
        fig = self.get('figure');
        if fig is None:
            fig = figure();
            self.set('figure',fig);
        if ax is None:
            ax = fig.add_subplot(131, projection='3d');
            self.set('globalAxes',ax);
            xlabel('x (km)');ylabel('y (km)');
            
        if grid:
            rs = arange(-1,2)*R;
            ax.plot(rs,rs-rs,rs-rs);ax.plot(rs-rs,rs,rs-rs);ax.plot(rs-rs,rs-rs,rs);
            
            lmds = arange(-90,90)*pi/180; chis = lmds - lmds;
            xs,ys,zs = self.lmdchis2xyz(lmds,chis);
            ax.plot(xs,ys,zs,'k');
            xs,ys,zs = self.lmdchis2xyz(lmds,chis+pi);
            ax.plot(xs,ys,zs,'k');
            
            chis = arange(-180,180)*pi/180; lmds = chis - chis; 
            xs,ys,zs = self.lmdchis2xyz(lmds,chis+pi);
            ax.plot(xs,ys,zs,'k');        
            
        ax.plot(xs,ys,zs,pstyle);
        #show();
        return;
        
    def plotLonlat(self,lons,lats,pstyle="-"):
        R = self.get('earthRadius');
        ax = self.get('localAxes');
        fig = self.get('figure');
        if fig is None:
            fig = figure();
            self.set('figure',fig);
        if ax is None:
            #fig = figure()
            ax = fig.add_subplot(122);
            ylabel('Latitude (degree)');xlabel('Longitude (degree)');
            self.set('localAxes',ax);
            grid(True);
        ax.plot(lons,lats,pstyle);
        #show();
        return;

    def plotScanOrbit(self,xscans,yorbits,pstyle="-"):
        fig = self.get('figure');
        ax = self.get('scanorbitAxes');
        if fig is None:
            fig = figure();
            self.set('figure',fig);
        if ax is None:
            #fig = figure()
            ax = fig.add_subplot(121);
            ylabel('Satellite Orbit (km)');xlabel('Sensor Scan (km)');
            self.set('scanorbitAxes',ax);
            grid(True);
        ax.plot(xscans,yorbits,pstyle);
        #show();
        return;
        
    def plot3(self,lmds,chis,pstyle="k-"):
        self.plotLonlat(chis/pi*180,lmds/pi*180,pstyle=pstyle);
        xs,ys,zs = self.lmdchis2xyz(lmds,chis);
#        self.plotXyz(xs,ys,zs,'k');
        xscans,yorbits,znormals = self.xyz2scanorbit(xs,ys,zs);
        self.plotScanOrbit(xscans,yorbits,pstyle=pstyle)
        return;
        
    def plotSpace(self,lata = None,lona =None,pstyle="k-"):
        if lata is None and lona is None:
            lata = self.get('latArray');
            lona = self.get('lonArray');
        Nrow, Ncol = lata.shape;
        
        self.plot3(lata[:,0]/180*pi,lona[:,0]/180*pi,pstyle=pstyle)
        self.plot3(lata[:,-1]/180*pi,lona[:,-1]/180*pi,pstyle=pstyle)
        self.plot3(lata[0,:]/180*pi,lona[0,:]/180*pi,pstyle=pstyle)
        self.plot3(lata[-1,:]/180*pi,lona[-1,:]/180*pi,pstyle=pstyle)
        
        self.plot3(lata[int(Nrow/2),:]/180*pi,lona[int(Nrow/2),:]/180*pi,pstyle=pstyle)
        self.plot3(lata[:,int(Ncol/2)]/180*pi,lona[:,int(Ncol/2)]/180*pi,pstyle=pstyle)
        
        return;
        

    def pre_findScanOrbitAxes(self):
        print('# pre_findScanOrbitAxes #');
        lata = self.get('latArray');
        lona = self.get('lonArray');
        Nrow, Ncol = lata.shape;
        self.set('Norbit',Nrow);
        self.set('Nscan',Ncol);
        lmda = lata/180*pi;
        lona = lona/180*pi;
        xa,ya,za = self.lmdchis2xyz(lmda,lona);
        xc = xa[int(Nrow/2),int(Ncol/2)];
        yc = ya[int(Nrow/2),int(Ncol/2)];
        zc = za[int(Nrow/2),int(Ncol/2)];
        
        # satellite scan axis
        xScans = xa[int(Nrow/2),:];
        yScans = ya[int(Nrow/2),:];
        zScans = za[int(Nrow/2),:];
        difxa_scan = diff(xa,axis=1)[:,int(Ncol/2)-1];
        difya_scan = diff(ya,axis=1)[:,int(Ncol/2)-1];
        difza_scan = diff(za,axis=1)[:,int(Ncol/2)-1];
        I = difxa_scan!=0;
        cy = mean(difya_scan[I]/difxa_scan[I]);
        cz = mean(difza_scan[I]/difxa_scan[I]);
        print('scan dx,dy,dz:',difxa_scan.mean(),difya_scan.mean(),difza_scan.mean());
        #cy = difya_scan[int(Nrow/2)]/difxa_scan[int(Nrow/2)];
        #cz = difza_scan[int(Nrow/2)]/difxa_scan[int(Nrow/2)];        
        scanAxis = array([1,cy,cz])/(1+cy**2+cz**2)**0.5;
        
        # satellite orbit axis
        xOrbits = xa[:,int(Ncol/2)];
        yOrbits = ya[:,int(Ncol/2)];
        zOrbits = za[:,int(Ncol/2)];
        difxa_orbit = diff(xa,axis=0)[int(Nrow/2)-1,:];
        difya_orbit = diff(ya,axis=0)[int(Nrow/2)-1,:];
        difza_orbit = diff(za,axis=0)[int(Nrow/2)-1,:];
        I = difxa_orbit!=0;
        cy = mean(difya_orbit[I]/difxa_orbit[I]);
        cz = mean(difza_orbit[I]/difxa_orbit[I]);
        print('orbit dx,dy,dz:',difxa_orbit.mean(),difya_orbit.mean(),difza_orbit.mean());
        #cy = difya_orbit[int(Ncol/2)]/difxa_orbit[int(Ncol/2)];
        #cz = difza_orbit[int(Ncol/2)]/difxa_orbit[int(Ncol/2)];
        #cy = (yOrbits[-1]-yOrbits[0])/(xOrbits[-1]-xOrbits[0]);
        #cz = (zOrbits[-1]-zOrbits[0])/(xOrbits[-1]-xOrbits[0]);
        orbitAxis = array([1,cy,cz])/(1+cy**2+cz**2)**0.5;     
        normalAxis = cross(scanAxis,orbitAxis);
        transMatrix = array([scanAxis,orbitAxis,normalAxis]);
    
        
#        print("axes dot:",dot(scanAxis,orbitAxis),dot(scanAxis,normalAxis),dot(orbitAxis,normalAxis));
#        print("center xyz:",xc,yc,zc);
        dscan = scanAxis*1000;
        dorbit = orbitAxis*1000;
        dnormal = normalAxis*1000;
#        self.plotXyz([xc],[yc],[zc],'o')
#        self.plotXyz([xc-dscan[0],xc,xc+dscan[0]],[yc-dscan[1],yc,yc+dscan[1]],[zc-dscan[2],zc,zc+dscan[2]],'>-');
#        self.plotXyz([xc-dorbit[0],xc,xc+dorbit[0]],[yc-dorbit[1],yc,yc+dorbit[1]],[zc-dorbit[2],zc,zc+dorbit[2]],'>-');
#        self.plotXyz([xc-dnormal[0],xc,xc+dnormal[0]],[yc-dnormal[1],yc,yc+dnormal[1]],[zc-dnormal[2],zc,zc+dnormal[2]],'>-');
        
        self.set('xyz2ScanorbitMatrix',transMatrix);
        self.set('scanorbitOrigin',array([xc,yc,zc]));
        self.set('scanRangeCartesan',array([xScans,yScans,zScans]));
        self.set('orbitRangeCartesan',array([xOrbits,yOrbits,zOrbits]));
        print('found axis:',transMatrix);
        
        return;

    
    def pre_findScanOrbitRange(self):
        print('# pre_findScanOrbitRange #');
        lata = self.get('latArray');
        lona = self.get('lonArray');
        Nrow, Ncol = lata.shape;
        lmda = lata/180*pi;
        chia = lona/180*pi;
        xa,ya,za = self.lmdchis2xyz(lmda,chia);
        xscana,yorbita,znormala = self.xyz2scanorbit(xa,ya,za);
        xscanMin = xscana.min();
        xscanMax = xscana.max();
        yorbitMin = yorbita.min();
        yorbitMax = yorbita.max();
        scanorbitRange = array([xscanMin,xscanMax,yorbitMin,yorbitMax]);
        nrmRange = array([znormala.min(),znormala.max()]);
        
        self.set('scanorbitRange',scanorbitRange);
        self.set('nrmRange',nrmRange);
        print('found scanorbitRange',scanorbitRange);
        print('found nrmRange',nrmRange);
        
        #self.pre_findScanOrbitDierctRange();
        return scanorbitRange;

    def pre_findScanOrbitDierctRange(self):
        # find scan and orbit axes in direct coordinates
        print('# pre_findScanOrbitDierctRange #');
        scanXyzs = self.get('scanRangeCartesan');
        orbitXyzs = self.get('orbitRangeCartesan');

        xscn,yscn,zscn = self.xyz2scanorbit(scanXyzs[0],scanXyzs[1],scanXyzs[2]);
        self.set('scanRangeDirect',xscn);
        
        xobt,yobt,zobt = self.xyz2scanorbit(orbitXyzs[0],orbitXyzs[1],orbitXyzs[2]);
        self.set('orbitRangeDirect',yobt);
                
        #figure();
        #subplot(1,2,1); plot(xscans,'o-');
        #subplot(1,2,2); plot(yorbits,'o-')
        return xscn,yobt;        
    
    def xscanYorbit2indices(self,scns,obts,nrms=None):
        from scipy import interpolate;
        scnRange = self.get('scanRangeDirect');
        obtRange = self.get('orbitRangeDirect');
        Jscan = arange(len(scnRange));
        Iorbit = arange(len(obtRange));
        
        
        scn_interp = interpolate.interp1d(scnRange,Jscan,bounds_error=False,fill_value=-100);
        obt_interp = interpolate.interp1d(obtRange,Iorbit,bounds_error=False,fill_value=-100);
        #iorbits = interp(obts,obtRange,Iorbit);
        #jscans = interp(scns,scnRange,Jscan);
        #print("type:",type(obts),type(array(obts)));
        iorbits = obt_interp(array(obts));
        jscans = scn_interp(array(scns));
#        print("mean nrms:",nrms.mean())
        
        nrmRange = self.get('nrmRange');
        if nrms is not None: #take care of the negative local z values
            I =  logical_or(nrms<=nrmRange[0],nrms>=nrmRange[1]);
            iorbits[I] = NaN;
            jscans[I] = NaN;
        return jscans,iorbits;

        
    def latlons2indices(self,lats,lons):
        lmds = lats*pi/180;
        chis = lons*pi/180;
        xs,ys,zs = self.lmdchis2xyz(lmds, chis);
        scns, obts, nrms = self.xyz2scanorbit(xs,ys,zs);
        
        jscans,iorbits = self.xscanYorbit2indices(scns,obts,nrms);
        return jscans,iorbits,scns,obts,xs,ys,zs;

    def latlonsTieConvert(self,nrowFull, ncolFull, lata,lona):
        nrow, ncol = lata.shape;
        scna,obta,nrma =self.lmdchis2scanorbit( lata/180*pi, lona/180.*pi);
        scnaNew = zeros([nrow, ncolFull]);
        obtaNew = zeros([nrow, ncolFull]);
        nrmaNew = zeros([nrow, ncolFull]);
        for i in range(0,nrow):
            scnaNew[i,:], difscnsNew = resizeMono(scna[i,:], ncolFull);
            obtaNew[i,:], difscnsNew = resizeMono(obta[i,:], ncolFull);
            nrmaNew[i,:], difscnsNew = resizeMono(nrma[i,:], ncolFull);
        scnaNew1 = zeros([nrowFull, ncolFull]);
        obtaNew1 = zeros([nrowFull, ncolFull]);
        nrmaNew1 = zeros([nrowFull, ncolFull]);    
        for j in range(ncolFull):
            scnaNew1[:,j], difscnsNew = resizeMono(scnaNew[:,j], nrowFull);
            obtaNew1[:,j], difscnsNew = resizeMono(obtaNew[:,j], nrowFull);
            nrmaNew1[:,j], difscnsNew = resizeMono(nrmaNew[:,j], nrowFull);
        return scnaNew1,obtaNew1,nrmaNew1

    def latlon2hvij(self, lat1, lon1):
        lat = 0 + lat1;
        lon = 0 + lon1;
        I = lon>180;
        lon[I] = lon[I]-180;
        ###Calculate tile h,v,i,j
        row_prj = -1*lat+90;
        col_prj = lon * np.cos(lat*pi/180)+180;
        ## find h&v
        h_indx = (col_prj/10).astype(int) ;
        v_indx = (row_prj/10).astype(int);
        i_indx = ((col_prj-h_indx*10)*240).astype(int);
        j_indx = ((row_prj-v_indx*10)*240).astype(int); 
        print('indx h', h_indx.min(), h_indx.max());
        print('indx v', v_indx.min(), v_indx.max());
        print('lon, lat', lon1.min(), lon1.max(), lat1.min(), lat1.max());
        return h_indx, v_indx, i_indx, j_indx
        
    def matchTile(self, lata, lona, varname, dirIn_tile):
        ha_indx, va_indx, ia_indx, ja_indx = self.latlon2hvij(lata, lona)
        nrow ,ncol =lata.shape;
        ja_g,ia_g = meshgrid(arange(ncol), arange(nrow));
        
        hmin_indx = ha_indx.min();
        hmax_indx = ha_indx.max();
        vmin_indx = va_indx.min();
        vmax_indx = va_indx.max();    
        var_new = lata-lata+255;
        count = 0;
        for ih in range(hmin_indx, hmax_indx+1):
            for iv in range(vmin_indx, vmax_indx+1):
                I = logical_and(ha_indx == ih, va_indx == iv);
                i_indx1 = ia_indx[I];
                j_indx1 = ja_indx[I];
                is_g = ia_g[I];
                js_g = ja_g[I];
                hnnvnn = 'h'+f'{ih:02}'+'v'+f'{iv:02}'
                print(hnnvnn)
                f_tiles = ReadFileNames(dirIn_tile, 'MCD12Q1*'+hnnvnn+'*.hdf')
                if len(f_tiles) > 0 :
                    print('tile name: ', hnnvnn)
                    f_tile = f_tiles[0];
                    vara = readHDF_data(f_tile,varname);
                    var_new[is_g,js_g] = vara[j_indx1, i_indx1];   
#                    pcolormesh(240*ih,240*iv, vara[0:-1:10,0:-1:10]);
#                    colorbar();
        return var_new

