import h5py
import numpy as np
import netCDF4 as nc
import glob
from os import path
from scipy import signal,integrate,stats
from datetime import date, timedelta, datetime, timezone
from obspy import read as oread
from obspy import UTCDateTime
import cmocean


# library of functions to read raw das data

# things to load for each file: 
# strain (m/m), time (UTC timestamp), channels, depth (m), metadata(gL, dx, fs)
# strain format is rows=time; cols=channels

# label lines by water depth
def all_line_info():
    # in order of Duck, KNO, MVCO, Oliktok, Homer, Florence
    h = [-6,-10,-13,15,-44,-50]
#     f_noise = [0.48,0.32,0.32,0.3,0.22,0.24]
    f_noise = [0.53,0.37,0.37,0.35,0.26,0.26]
    cmap = cmocean.cm.deep
    clrs = cmap(np.linspace(0, 1, 6))
    clrs[0,2]=0.7

    # v1 line colors 
#     x = np.linspace(0.0, 1.0, 7)
#     clrs = matplotlib.colormaps['ocean_r'](x)
#     clrs = clrs[1:,:]

    # v0 version of line colors:
#     x = np.linspace(0.0, 1.0, 45)
#     clrs = matplotlib.colormaps['ocean'](x)
#     x = np.arange(-51,-6,1)
    return h, f_noise, clrs


# Case 1: Duck
# das files are saved in 17 min chunks with all channels 
# depth and FRF x/y are saved in channelXYZ
def loadDuck(pname,fname):
    f = h5py.File(pname+fname, 'r')

    # load data and metadata from the h5 file
    das_time = f['Acquisition']['Raw[0]']['RawDataTime'][:150000]/1e6
    rawstrain = f['Acquisition']['Raw[0]']['RawData'][:150000,:]
    fs = f['Acquisition']['Raw[0]'].attrs['OutputDataRate'] # sampling rate in Hz
    dx = f['Acquisition'].attrs['SpatialSamplingInterval'] # channel spacing in m
    nx = f['Acquisition']['Raw[0]'].attrs['NumberOfLoci'] # number of channels
    L = f['Acquisition'].attrs['GaugeLength'] # gauge length in m
    n = 1.46 # fp['Acquisition']['Custom'].attrs['Fibre Refractive Index'] # refractive index
    rawstrain = rawstrain * ((1550.12 * 1e-9)/(0.78 * 4 * np.pi * n * L))

    # dictionary of metadata
    metadata = {'fs':fs,'gl':L,'dx':dx} #metadata['fs']

    # define channels and depth
    chnl = np.arange(0,nx)
    fname = 'channelXYZ.csv'
    f = np.genfromtxt(pname+fname, delimiter=',')
    depth = f[:,2]
    return rawstrain,das_time,chnl,depth,metadata

# Case 2: KNO (fibers 1 and 2)
# das files are saved in 5 min chunks with all channels 
# approx cable depth for each fiber is saved in estimatedcabledepth.npz
# Sampling rate changes for both fibers. Fiber1=coarse gl. Fiber2=fine gl.
def loadKNO(pname,fname,fiber):
    f = h5py.File(pname+fname, 'r')

    # load data and metadata from the h5 file
    rawstrain = f['Acquisition']['Raw[0]']['RawData'][:]
    das_time = f['Acquisition']['Raw[0]']['RawDataTime'][:]/1e6
    fs = f['Acquisition']['Raw[0]'].attrs['OutputDataRate'] # sampling rate in Hz
    dx = f['Acquisition'].attrs['SpatialSamplingInterval'] # channel spacing in m
    nx = f['Acquisition']['Raw[0]'].attrs['NumberOfLoci'] # number of channels
    L = f['Acquisition'].attrs['GaugeLength'] # gauge length in m
    n = 1.46 # fp['Acquisition']['Custom'].attrs['Fibre Refractive Index'] # refractive index
    rawstrain = rawstrain * ((1550.12 * 1e-9)/(0.78 * 4 * np.pi * n * L))

    # dictionary of metadata
    metadata = {'fs':fs,'gl':L,'dx':dx} #metadata['fs']

    # define channels and depth
    chnl = np.arange(0,nx)

    # load estimated water depth
    with np.load(pname+'estimatedcabledepth.npz') as f:
        if fiber==1:
            depth = f['h_f1'][:]*(-1)
        elif fiber==2:
            depth = f['h_f2'][:]*(-1)
    return rawstrain,das_time,chnl,depth,metadata

# Case 3: MVCO
def loadMV(pname,fname):
    metadata = {'fs':2,'gl':10,'dx':4}
    d = nc.Dataset(pname+fname)
    samples = d['sample'][:]# calculated every 30 minutes
    chnl = d['channel'][:]
    rawstrain = d['strain'][:]*(1e-9) # convert from nanostrain rate to strain rate
    rawstrain = rawstrain.T
    # convert from strain rate to strain
#     rawstrain = (rawstrain[2:,:] - rawstrain[:-2,:])*(1/metadata['fs'])
#     rawstrain = np.insert(rawstrain,0, rawstrain[0,:],axis=0)
#     rawstrain = np.insert(rawstrain,-1, rawstrain[-1,:],axis=0)
    #define time based on file name
    Y = int(fname[0:4])
    M = int(fname[4:6])
    D = int(fname[6:8])
    H = int(fname[8:10])
    t = datetime(Y, M, D, H, 0, 0,tzinfo=timezone.utc).timestamp()
    das_time = t+np.arange(0,30*60,0.5)
    

    return rawstrain,das_time,chnl,metadata

# case 4: Oliktok, AK (adapted from Maddie and Jim)
# das files are saved in 15 second chunks by channel
# approx cable depth for each fiber is saved in CODAS_info.csv
def loadOliktok(pname,chnl,t1,t2,convert_strain):
    # load info about cable path (#,indx,lat,lon,dist,depth,channel)
    ff = np.genfromtxt(pname+'CODAS_info.csv', delimiter=',',skip_header=1)
    Xdist = ff[:,4]
    depth = (-1)*ff[:,5] #define negative down
    chn_all = ff[:,6]

    Xdist = np.interp(chnl,chn_all,Xdist)
    depth = np.interp(chnl,chn_all,depth)
    
    #get all times from file names for 1 channel
    fnames = glob.glob(pname+'rawData/'+str(chnl[0])+'/*.sac')
    file_time = sorted([path.basename(x) for x in fnames])
    file_timestr = np.asarray(sorted([x[10:25] for x in file_time]))
    file_time = np.asarray([datetime.strptime(x[10:25],'%Y%m%d_%H%M%S').timestamp() for x in file_time])
    
    # only load relevant time chunk (too much data to load all times)
    file_timestr = file_timestr[(file_time>=t1) & (file_time<=t2)]
    file_time = file_time[(file_time>=t1) & (file_time<=t2)]
    # hard code metadata:
    fs = 100
    metadata = {'fs':fs,'gl':10,'dx':20} 
    
    das_time = []
    for ci in chnl:    
        # load all data for this channel from 15 second file segments
        strain_rate = []
        for ti in range(len(file_time)):
            das_file = (pname+'rawData/'+str(ci)+'/CODAS.D*' + file_timestr[ti] + '*.sac')
            ds = oread(das_file) 
            strain_rate = np.append(strain_rate,ds[0].data) #check shape of ds.data - want it to be 1 column, many rows
            if ci==min(chnl): # load time with first channel
                t = file_time[ti] + np.arange(0,len(ds[0].data)*(1/fs),1/fs)
                das_time = np.append(das_time,t) #check actual name of time field in netcdf
        if ci==min(chnl):
            rawstrain = strain_rate
        else:
            rawstrain = np.vstack([rawstrain,strain_rate]) #convert to strain from strain rate
    rawstrain = rawstrain.T * 1e-9 # data was saved in nano-strainrate (nm/m/s) -> convert back to strain rate (m/m/s)
    das_time -= (8*60*60)
    
    #convert to strain from strain rate
    if convert_strain==1:
        rawstrain = (rawstrain[2:,:] - rawstrain[:-2,:])*(1/metadata['fs'])
        rawstrain = np.insert(rawstrain,0, rawstrain[0,:],axis=0)
        rawstrain = np.insert(rawstrain,-1, rawstrain[-1,:],axis=0)

    
    return rawstrain,das_time,chnl,depth,Xdist,metadata

# case 4: Oliktok, AK single file
def loadOliktokSingle(pname,fname):
    # load info about cable path (#,indx,lat,lon,dist,depth,channel)
    ff = np.genfromtxt(pname+'CODAS_info.csv', delimiter=',',skip_header=1)
    chn_all = ff[:,6]
    depth = int((-1)*ff[chn_all==10840,5]) #define negative down
    fs = 100
    metadata = {'fs':fs,'gl':10} 
    ds = oread(pname+fname)
    rawstrain = ds[0].data
    rawstrain = rawstrain.T * 1e-9 # data was saved in nano-strainrate (nm/m/s) -> convert to microstrain rate (mm/m/s)
#     rawstrain = (rawstrain[1:] - rawstrain[:-1])*(1/fs) #convert to strain from strain rate
    file_time = datetime.strptime('20211110_170007-UTC','%Y%m%d_%H%M%S-%Z').timestamp()
    das_time = file_time + np.arange(0,len(ds[0].data)*(1/fs),1/fs)
    rawstrain = rawstrain[:(fs*60*5)]
    das_time = das_time[:(fs*60*5)]

#     rawstrain = (rawstrain[2:] - rawstrain[:-2])*(1/metadata['fs'])
#     rawstrain = np.insert(rawstrain,0, rawstrain[0])
#     rawstrain = np.insert(rawstrain,-1, rawstrain[-1])

    return rawstrain,das_time,depth,metadata


# case 5: Homer, AK 
def loadHomer(pname,fname,onechn):
    if onechn==True:
        chnl = 5
        with h5py.File(pname+fname,'r') as fp:
            rawstrain = fp['RawData'][:,chnl]
            das_time = fp['RawDataTime'][:]/1e6
            depth = fp['WaterDepth'][chnl]
            dx = fp['RawData'].attrs['ChannelPitch']
            fs = fp['RawData'].attrs['SamplingRate']
            L = 17.5476194762 # gauge length, forgot to put it in the file
        # dictionary of metadata
        metadata = {'fs':fs,'gl':L} 
        # convert phase to strain
        rawstrain = rawstrain * ((1550.12 * 1e-9)/(0.78 * 4 * np.pi * 1.46 * L)) 
        #convert to strain from strain rate
#         rawstrain = (rawstrain[2:] - rawstrain[:-2])*(1/metadata['fs'])
#         rawstrain = np.insert(rawstrain,0, rawstrain[0])
#         rawstrain = np.insert(rawstrain,-1, rawstrain[-1])

        idx = (das_time>datetime.fromisoformat('2023-06-19 21:00:00').timestamp()) & (das_time<datetime.fromisoformat('2023-06-19 21:06:00').timestamp())
        rawstrain = rawstrain[idx]
        das_time = das_time[idx]
    else:    
        with h5py.File(pname+fname,'r') as fp:
            rawstrain = fp['RawData'][:]
            das_time = fp['RawDataTime'][:]/1e6
            depth = fp['WaterDepth'][:]
            dx = fp['RawData'].attrs['ChannelPitch']
            fs = fp['RawData'].attrs['SamplingRate']
            L = 17.5476194762 # gauge length, forgot to put it in the file
        # dictionary of metadata
        metadata = {'fs':fs,'gl':L,'dx':dx}
        # convert phase to strain
        rawstrain = rawstrain * ((1550.12 * 1e-9)/(0.78 * 4 * np.pi * 1.46 * L))
        # convert to strain from strain rate
#         rawstrain = (rawstrain[2:,:] - rawstrain[:-2,:])*(1/metadata['fs'])
#         rawstrain = np.insert(rawstrain,0, rawstrain[0,:],axis=0)
#         rawstrain = np.insert(rawstrain,-1, rawstrain[-1,:],axis=0)

        chnl = np.arange(0,len(depth))
         
        
    return rawstrain,das_time,chnl,depth,metadata

# Case 6: Florence
# das files are saved in 5 min chunks with all channels 
# approx cable depth for each fiber is saved in ???
# Sampling rate changes
def loadFlorence(pname,fname):
    f = h5py.File(pname+fname, 'r')

    # load data and metadata from the h5 file
    rawstrain = f['Acquisition']['Raw[0]']['RawData'][:,:800]
    das_time = f['Acquisition']['Raw[0]']['RawDataTime'][:]/1e6
    fs = f['Acquisition']['Raw[0]'].attrs['OutputDataRate'] # sampling rate in Hz
    dx = f['Acquisition'].attrs['SpatialSamplingInterval'] # channel spacing in m
    nx = 800 # number of channels
    L = f['Acquisition'].attrs['GaugeLength'] # gauge length in m
    n = 1.46 # fp['Acquisition']['Custom'].attrs['Fibre Refractive Index'] # refractive index
    rawstrain = rawstrain * ((1550.12 * 1e-9)/(0.78 * 4 * np.pi * n * L))

    # dictionary of metadata
    metadata = {'fs':fs,'gl':L,'dx':dx} #metadata['fs']

    # define channels and depth
    chnl = np.arange(0,nx)

    # load estimated water depth
    depth = np.linspace(-2, 50, num=nx)*(-1)

    return rawstrain,das_time,chnl,depth,metadata




#### library of funcs to do wave stuff with DAS and ground truth

def dispersion(h,T):
    # %
    # % [L,error,count] = dispersion(h,T);
    # % h=water depth(m), T=wave period (s)
    # % Numerically solves the linear dispersion relationship for water waves. 
    # % omega^2=gk*tanh(kh); where omega = 2*pi/T; k=2*pi/L; g=9.81;
    # % T and h provided as inputs.  Can handle T and/or h as vector inputs, for
    # % which the resulting size is L(m,n) for h(1,m) and T(1,n).
    # %
    # % Iterates using Newton-Raphson initialized with Guo (2002), which
    # % converges in under 1/2 the iterations of starting with deep water.
    # %
    # % Returns the error array and iteration count along with the wavelength.
    # %
    g = 9.81;
    omega = 2 * np.pi / T
    h = np.abs(h)

    # initial guess
    k = omega ** 2 / g * (1 - np.exp(-(omega * np.sqrt(h / g)) ** (5/2))) ** (-2/5)

    # iterate until error converges
    count = 0
    error = 1
    while np.any(error > (10 * np.finfo(float).eps)):
        f = omega ** 2 - g * k * np.tanh(k * h)
        dfdk = - g * np.tanh(k * h) - g * h * k * (1 / np.cosh(k * h)) ** 2
        k1 = k - f / dfdk
        error = abs((k1 - k) / k1)
        k = k1
        count += 1

    # wavelength
    L = 2 * np.pi / k;
    
    return L,k



def DAS_wave_conversion(das_data,fs,depth,strain_fac,strain_fac_frq,f_cutoff):
    #function to use simple pwelch to estimate wave spectra and bulk wave parameters for DAS data
    depth = np.abs(depth)
    nperseg = 512
    f_psd, ds_psd = signal.welch(das_data,fs=fs,nperseg=nperseg)

    # correct from PSD of strain to PSD of pressure
    C = np.interp(f_psd,strain_fac_frq,strain_fac)
    ds_psd *= C
    
    # cut off data above a precalculated noise floor frequency
    ds_psd[f_psd>f_cutoff] = np.nan
    
    # translate bed to surface
    f_psd[f_psd==0] = 0.001
    attenuation = calc_attenuation(f_psd,depth)
    ds_psd_corr = ds_psd*attenuation
    
    if np.isfinite(ds_psd_corr).any():
        # fill noise floor with f^-4 extrapolation:
        f_noise = f_psd[np.isnan(ds_psd_corr)]
        min_psd = np.min(ds_psd_corr[(f_psd>0.1) & (~np.isnan(ds_psd_corr))])
        ds_psd_corr[np.isnan(ds_psd_corr)] = (min_psd)*np.power(f_noise[0],4)*np.power(f_noise,-4)

        #calculate bulk wave characteristics
        ds_psd_corr = ds_psd_corr[(f_psd > 0.04) & (f_psd < 0.4)]
        f_psd = f_psd[(f_psd > 0.04) & (f_psd < 0.4)]
        fe = ((ds_psd_corr * f_psd) / ds_psd_corr.sum() ).sum() #(f*E)/E
        Te = 1/fe
        Tp = 1/(f_psd[np.argmax(ds_psd_corr)])

        bandwidth = (f_psd[1::] - f_psd[0:-1]).mean()
        Hs = 4*np.sqrt( ds_psd_corr.sum() * bandwidth )
    else:
        Tp = np.nan
        Te = np.nan
        Hs = np.nan

    return Tp, Te, Hs

# function adapted from mms to calculate Hs, Tp, and Te from spectra of pressure in meters
def pres_wave_conversion(pressure,fs,depth):
    #function to use simple pwelch to estimate wave spectra and bulk wave parameters 
    #pwelch - defualt is 50% overlap hanning window
    depth = np.abs(depth)
    nperseg = 512
    f_psd, ds_psd = signal.welch(pressure,fs=fs,nperseg=nperseg)    
    
    # translate bed to surface
    if depth==0:
        attenuation = 1
    else:
        attenuation = calc_attenuation(f_psd,depth)
        ds_psd_corr = ds_psd*attenuation
    
    # fill noise floor with f^-4 extrapolation:
    f_noise = f_psd[np.isnan(ds_psd_corr)]
    min_psd = np.min(ds_psd_corr[(f_psd>0.1) & (~np.isnan(ds_psd_corr))])
    ds_psd_corr[np.isnan(ds_psd_corr)] = (min_psd)*np.power(f_noise[0],4)*np.power(f_noise,-4)
    
    #calculate bulk wave characteristics
    ds_psd_corr = ds_psd_corr[(f_psd > 0.04) & (f_psd < 0.4)]
    f_psd = f_psd[(f_psd > 0.04) & (f_psd < 0.4)]
    fe = ((ds_psd_corr * f_psd) / ds_psd_corr.sum() ).sum() #(f*E)/E
    Te = 1/fe
    Tp = 1/(f_psd[np.argmax(ds_psd_corr)])

    bandwidth = (f_psd[1::] - f_psd[0:-1]).mean()
    Hs = 4*np.sqrt( ds_psd_corr.sum() * bandwidth ) 
    
#     return f_psd, ds_psd, ds_psd_corr, Tp, Te, Hs
    return Tp, Te, Hs


# Gauge Length effects
def gaugeLengthEffect(gl,C,T):
    # gl = gauge length
    # C = array or value with speeds that youre interested in (e.g., 1500 m/s for sound or 1400:1600 m/s)
    # T = range of periods youre interested in
    # The gauge length to wavelength ratio is accounted for in Hubbard eqn 11,
    # G = sin(pi*k*gl) / (pi*k*gl)
    # When G = 1, the gauge length does not impact the recorded strain.
    # When G = 0, the gauge length is some multiple of the wavelength and no strain is recorded.

    # pre-allocate empty array
    G = np.empty([len(C),len(T)])*np.nan

    for jj in range(len(C)):
        # calculate wavenumber based on sound speed and period
        k = 1 / (C[jj] * T)
        # calculate G
        G[jj,:] = np.abs(np.sin(np.pi*k*gl) / (np.pi*k*gl))
    
    return G

# Function adapted from https://github.com/leabouffaut/DAS4Whales
def fk_filter_design(trace_shape, dx, fs, cs_min, cp_min, cp_max, cs_max):
    """
    Designs a f-k filter for DAS strain data
    Keeps by default data with propagation speed [1450-3400] m/s

    The transition band is inspired and adapted from Yi Lin's matlab fk function
    https://github.com/nicklinyi/seismic_utils/blob/master/fkfilter.m

    Inputs:
    :param trace_shape: a tuple with the dimensions of the strain data in the spatio-temporal domain such as
    trace_shape = (trace.shape[0], trace.shape[1]) where dimensions are [channel x time sample]
    #HEG switched time/space axes, originally [space,time], now [time,space]
    :param dx: the channel spacing (m)
    :param fs: the sampling frequency (Hz)
    :param cs_min: the minimum selected sound speeds for the f-k passband filtering (m/s). Default 1400 m/s
    :param cp_min: the minimum selected sound speed for the f-k stopband filtering, values should frame
    [c_min and c_max] (m/s). Default 1450 m/s.
    :param cp_max: the maximum selected sound speeds for the f-k passband filtering (m/s). Default 3400 m/s
    :param cs_max: the maximumselected sound speed for the f-k stopband filtering, values should frame
    [c_min and c_max] (m/s). Default 3500 m/s

    Outputs:
    :return: fk_filter_matrix, a [channel x time sample] nparray containing the f-k-filter

    """

    # Note that the chosen ChannelStep limits the bandwidth frequency obtained with fmax = 1500/ChannelStep*dx

    # Get the dimensions of the trace data #HEG switched time/space axes, originally [space,time]
    nns, nnx = trace_shape

    # Define frequency and wavenumber axes
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=1 * dx))

    # Supress/hide the warning
    np.seterr(invalid='ignore')

    # Create the filter
    # Wave speed is the ratio between the frequency and the wavenumber
    fk_filter_matrix = np.ndarray(shape=(len(freq), len(knum)), dtype=float, order='F')

    # Going through wavenumbers
    for i in range(len(knum)):
        # Taking care of very small wavenumber to avoid 0 division
        if abs(knum[i]) < 0.005:
            fk_filter_matrix[:, i] = np.zeros(shape=[len(freq)], dtype=float, order='F')
        else:
            filter_line = np.ones(shape=[len(freq)], dtype=float, order='F')
            speed = abs(freq / knum[i])

            # Filter transition band, ramping up from cs_min to cp_min
            selected_speed_mask = ((speed >= cs_min) & (speed <= cp_min))
            filter_line[selected_speed_mask] = np.sin(0.5 * np.pi *
                                                      (speed[selected_speed_mask] - cs_min) / (cp_min - cs_min))
            # Filter transition band, going down from cp_max to cs_max
            selected_speed_mask = ((speed >= cp_max) & (speed <= cs_max))
            filter_line[selected_speed_mask] = 1 - np.sin(0.5 * np.pi *
                                                          (speed[selected_speed_mask] - cp_max) / (cs_max - cp_max))
            # Stopband
            filter_line[speed >= cs_max] = 0
            filter_line[speed < cs_min] = 0

            # Fill the filter matrix
            fk_filter_matrix[:, i] = filter_line

    return fk_filter_matrix

def fk_filter_filt(trace, fk_filter_matrix):
    """
    Applies a pre-calculated f-k filter to DAS strain data

    Inputs:
    :param trace: a [channel x time sample] nparray containing the strain data in the spatio-temporal domain
    #HEG switched time/space axes, originally [space,time], now [time,space]
    :param fk_filter_matrix: a [channel x time sample] nparray containing the f-k-filter

    Outputs:
    :return: trace, a [channel x time sample] nparray containing the f-k-filtered strain data in the spatio-temporal
    domain

    """

    # Calculate the frequency-wavenumber spectrum
    fk_trace = np.fft.fftshift(np.fft.fft2(trace))

    # Apply the filter
    fk_filtered_trace = fk_trace * fk_filter_matrix

    # Back to the t-x domain
    trace = np.fft.ifft2(np.fft.ifftshift(fk_filtered_trace))

    return trace.real

def dynpres(rho, H, L, h, omega, z, t, x):
# dynamic pressure
# uses linear wave theory to calculate the dynamic
# pressure at intermediate depths
# rho = density of water
# H = wave height, L = wavelength, h = water depth, omega = angular frequency (2pi/T)
# z = position in water column want pressure, still water surface = 0, positive upward
# x = points at same spatial resolution as h
    g=9.8 # gravity m/s2
    x -= x[0]
#     x = np.arange(0, L*3, .32)
    a = H / 2
    k = 2 * np.pi / L
    p_d = (rho * g * a * (np.cosh(k * (z + h)) / np.cosh(k * h))* np.sin(omega * t - k * x))
    
    dp = np.gradient(p_d)
    dx = np.gradient(x)
    dpdx = dp / dx
    return p_d, dpdx

# calculate spectra of das and translate to surface spectra
def surfaceSpec(rawstrain, fs, h, f_noise):    
    h = np.abs(h)
    frq,psd = signal.welch(rawstrain,fs=fs,window='hann',nperseg=fs*60,detrend=False)
    # Calculate depth attenuation function 
    frq[0] = frq[1]
    
    attenuation = calc_attenuation(frq,h)
    psd = psd*attenuation
#    psd = 20*np.log10(psd); # dB rel uE
    psd[frq>f_noise]=np.nan
    return frq, psd
    
# interrogator correction factor (doesnt deal with cable composition)
# used to compare spectra with gauge length effects removed
def interr_corr(rawstrain, h, gl, pw, ns, fs):
    # % gl = gauge length (m);
    # % pw = pulse width in seconds;
    # % ns = number of samples in time series;
    # % fs = sampling frequency (Hz);
    # % h = water depth as a positive value (m)
    h = np.abs(h)
    g = 9.8 #gravity in m/s2

    # convert to frequency space
    frq = np.fft.rfftfreq(100,d=1)
    frq[0] = frq[1] # avoid 0 in calculating k below

    # calculate wavenumber from dispersion relationship for the range of frequencies sampled
    lamda, _ = dispersion(h,1/frq)
    k = 1 / lamda # wavenumber in 1/m
    # calculate G, gauge length effect, Hubbard 2022, eqn 11    
    G = np.abs(np.sin(np.pi*k*gl) / (np.pi*k*gl))

    # calculate P, pulse characteristics, Hubbard 2022, eqn 13
    hp = pw*299792458/1.46 # pulse width = 20 ns * C / n
    sigma_k = 2.3548/(2*np.pi*hp)
    P = np.exp((-k**2)/(2*sigma_k**2))

    # Input constant alpha - Reinsch 2017, eqn 8
    # Calculate H(k) = H_k = G * P * beta 
    # omit effects of cable composition
    H_k = G * P

    # interpolate H_k for the specific time series
    E = np.fft.rfft(rawstrain,n=ns,norm='ortho')
    f_E = np.fft.rfftfreq(ns,d=1/fs)
    H_k = np.interp(f_E,frq,H_k)
    
    P_k = E / H_k #not the full conversion here, still just strain, but normalized by cable effects

    # convert back to time series, specify variable length and normalization just for mental comfort
    newstrain = np.fft.irfft(P_k,n=ns,norm='ortho')

    return newstrain, H_k, frq

# another version of an interrogator correction factor (doesnt deal with cable composition)
# compare spectra with gauge length effects removed
def interr_corr2(rawstrain, h, gl, pw, ns, fs,f_noise):
    
    h = np.abs(h)
    # calc psd of strain with 60s windows
    frq,psd = signal.welch(rawstrain,fs=fs,window='hann',nperseg=fs*60,detrend=False)
    
    # calculate wavenumbers
    frq[0] = frq[1]
    L,_ = dispersion(h,1/frq)
    k = 1/L    
    
    # calculate G, gauge length effect, Hubbard 2022, eqn 11    
    G = np.abs(np.sin(np.pi*k*gl) / (np.pi*k*gl))

    # calculate P, pulse characteristics, Hubbard 2022, eqn 13
    hp = pw*299792458/1.46 # pulse width = 20 ns * C / n
    sigma_k = 2.3548/(2*np.pi*hp)
    P = np.exp((-k**2)/(2*sigma_k**2))

    # Combine to find H(k) ->1 means strain is fully recorded (flip for calculation of correction)
    # Calculate H(k) = H_k = G * P * alpha  (Reinsch 2017, eqn 8)
    # omit effects of cable composition
    H_k = 1 - (G * P)
#     H_k[frq>0.4] = np.nan

    # calculate corrected psd of strain
    # not the full conversion to pressure, still just strain, but normalized by cable effects
    psd *= (1 + H_k)
    
    # Account for depth attenuation-> surface spectra
    attenuation = calc_attenuation(frq,h)
    psd = psd*attenuation
    # convert back to time series, specify variable length and normalization just for mental comfort
#     newstrain = np.fft.irfft(P_k,n=ns,norm='ortho')
    
#    psd = 20*np.log10(psd); # dB rel uE
    psd[frq>f_noise]=np.nan

    
    return psd, frq, H_k


def calcnoisefloor(rawstrain,metadata):
    # calculate 1 dof FFT
    ns = len(rawstrain)
    frq = np.fft.rfftfreq(ns,d=1./metadata['fs']) # frequency in Hz
    psd = 20 * np.log10( (2/ns) * abs(np.fft.rfft(rawstrain * np.hamming(ns))) )

    # bin the data by frequency to find a rough noise floor
#     strt = frq[np.argmax(psd)]+0.05
    frq_bins = np.arange(0.2,1,0.02)
    psd_bins, _, _ = stats.binned_statistic(frq,psd, 'median', bins=frq_bins)
    # calculate the slope of the PSD between bins
    slp = np.absolute(psd_bins[:-1]-psd_bins[1:])
    # calculate the variability in each bin
    std_bins, _, _ = stats.binned_statistic(frq,psd, 'std', bins=frq_bins)
    std_bins = np.mean(std_bins)/5
    # if slope is less than STD then we've hit noise floor
    frq_noise = [ n for n,i in enumerate(slp) if i<std_bins][0]
    frq_noise = frq_bins[frq_noise]

    return frq_noise

def calc_attenuation(frq,depth):
    _,k = dispersion(depth,1/frq)
    attenuation = np.exp(k*depth)**2
    attenuation[attenuation>500]=np.nan
    return attenuation
