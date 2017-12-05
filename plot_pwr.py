#!/usr/bin/env python2.7
# -*- coding: iso-8859-1 -*-

# Extract data from SDM for plotting RFI
# This script was inspired by Casey Law who performed a similar analysis for VLASS a couple 
# years ago, some of his code was reused here. However it is now compatible to only require 
# sdmpy to interacting with the SDM/BDF, which is mainted by Paul Demorest.
# -- Frank Schinzel (10/06/2017)

import sdmpy
import sdmpy.bintab
import ephem
import pickle
import os
from contextlib import closing
import multiprocessing as mp
import numpy as n
import sys
import math
import astropy
import astropy.stats

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pylab as p
from mpl_toolkits.mplot3d import Axes3D

def read_metadata(sdmname):
  sdm = sdmpy.SDM(sdmname)
  scandict = {}; sourcedict={}

  idx = 0
  sidx = 0
  for scan in sdm.scans():
    ss = sdm.scan(scan.idx,scan.subidx)
    scandict[idx] = {}
    scandict[idx]['source'] = "%s" % ss.field
    scandict[idx]['startmjd'] = ss.startMJD
    scandict[idx]['endmjd'] = ss.endMJD
    scandict[idx]['intent'] = ss.intents
    scandict[idx]['scan'] = scan.idx
    scandict[idx]['subscan'] = scan.subidx
    scandict[idx]['duration'] = ss.endMJD-ss.startMJD
    scandict[idx]['nints'] = ss.numIntegration
    scandict[idx]['bdfstr'] = ss.bdf_fname

    if scandict[idx]['source'] not in [sourcedict[source]['source'] for source in sourcedict.iterkeys()]:
      sourcedict[sidx] = {}
      sourcedict[sidx]['source'] = "%s" % ss.field
      sourcedict[sidx]['ra'] = ss.coordinates[0]
      sourcedict[sidx]['dec'] = ss.coordinates[1]
      sidx += 1
    
    idx+=1

  return [scandict, sourcedict]

def read_amp(sdmname,scanlist,sc,nthread=4,intchunk=2):
  # iterate through scans and build RFI summaries
  rfilist = []
  results = []
  dim0 = 0
  with closing(mp.Pool(nthread)) as readpool:
    for scan in scanlist:
      for nskip in range(0, sc[scan]['nints']-intchunk+1, intchunk):
	results.append(readpool.apply_async(readreduce, [sdmname, scan, nskip, sc, intchunk]))

    for result in results:
      rfi = result.get()
      dim0 += len(rfi)
      rfilist.append(rfi)

  rfiarray = n.array(rfilist).squeeze()
  dim1 = rfiarray.shape[1]
  dim2 = rfiarray.shape[2]
  rfiarray = n.reshape(rfiarray,(dim0,dim1*dim2))

  return rfiarray

def readreduce(sdmname, scan, nskip, sc, intchunk):
  def reducedata(data):
    return n.abs(data).max(axis=4).max(axis=1).max(axis=0)[None,:] # (time, baseline, spw, chan, poln) 

  sdm = sdmpy.SDM(sdmname)
  scan = sdm.scan(sc[scan]['scan'],sc[scan]['subscan'])
  data = scan.bdf.get_data(trange=(nskip,nskip+intchunk))
  data = n.squeeze(data)

  return reducedata(data)

def get_spwfreq(sdmname,sc):
  sdm = sdmpy.SDM(sdmname)
  last_scan = sc.keys()[-1]
  scan = sdm.scan(sc[last_scan]['scan'],sc[last_scan]['subscan'])
  spwfreq = scan.reffreqs
  spwfreq = n.array(spwfreq)
  spwfreq = (spwfreq + 64.0e6)/1e9
  return spwfreq

def get_scanlist(sc,intent):
  scanlist = []
  for idx in range(len(sc)):
    if intent!="":
      if intent in sc[idx]['intent']:
	scanlist.append(idx)

    else:
      last_scan = sc.keys()[-1]
      scanlist = range(1,last_scan,1)
      break
  return scanlist

def calcaltaz(scanlist, sc, sr, format='str', intchunk=2):
    """ Calculates a single (alt,az) per scan in scanlist.
    """

    # inttime = (sc[1]['endmjd'] - sc[1]['startmjd'])*24*3600/sc[1]['nints']

    vla = ephem.Observer()
    vla.lat = '34:04:43.497'
    vla.long = '-107:37:03.819'
    vla.elevation = 2124
    src = ephem.FixedBody()


    altaz = []
    for scan in scanlist:
	inttime = (sc[scan]['endmjd'] - sc[scan]['startmjd'])*24*3600/sc[1]['nints']
        src._ra, src._dec = [(sr[srn]['ra']/360.0*24.0, sr[srn]['dec']) for srn in sr.keys() if sc[scan]['source'] == sr[srn]['source']][0]
        for nskip in range(0, sc[scan]['nints']-intchunk+1, intchunk):
            vla.date = ephem.date(jd_to_date(sc[scan]['startmjd'] + nskip*inttime/(24*3600) + 2400000.5))
            src.compute(vla)
            if format == 'str':
                altaz.append( '(%.1f, %.1f)' % (n.degrees(src.alt), n.degrees(src.az)) )
            elif format == 'float':
                altaz.append( (n.degrees(src.alt), n.degrees(src.az)) )

    return n.array(altaz)

def calcradec(scanlist, sc, sr, intchunk=2):
    radec = []
    for scan in scanlist:
        for nskip in range(0, sc[scan]['nints']-intchunk+1, intchunk):
            radec.append([(n.degrees(sr[srn]['ra']), n.degrees(sr[srn]['dec'])) for srn in sr.keys() if sc[scan]['source'] == sr[srn]['source']][0])

    return n.array(radec)

def jd_to_date(jd):
    """
    Convert Julian Day to date.
    
    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet', 
        4th ed., Duffet-Smith and Zwart, 2011.
    
    Parameters
    ----------
    jd : float
        Julian Day
        
    Returns
    -------
    year : int
        Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.
        
    month : int
        Month as integer, Jan = 1, Feb. = 2, etc.
    
    day : float
        Day, may contain fractional part.
        
    Examples
    --------
    Convert Julian Day 2446113.75 to year, month, and day.
    
    >>> jd_to_date(2446113.75)
    (1985, 2, 17.25)
    
    """
    jd = jd + 0.5

    F, I = math.modf(jd)
    I = int(I)

    A = math.trunc((I - 1867216.25)/36524.25)

    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I

    C = B + 1524

    D = math.trunc((C - 122.1) / 365.25)

    E = math.trunc(365.25 * D)

    G = math.trunc((C - E) / 30.6001)

    day = C - E + F - math.trunc(30.6001 * G)

    if G < 13.5:
        month = G - 1
    else:
        month = G - 13

    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715

    return year, month, day

# Match switched power data to scans
def read_swpw(sdmname):
  import sdmpy.bintab
  s = sdmpy.SDM(sdmname)
  sp = sdmpy.bintab.unpacker(s['SysPower'])
  sp.unpack()
  return sp

def select_swpw(swpw,sc,scanlist,intchunk=2,normscan=4):

  # create dictionary matched to scan 
  #i = 0
  #j = 0
  #swpw = {}

  ## Fill dictionary with switched power values
  #for row in sp.row:
    ## data format of row: ('Antenna_21', 'SpectralWindow_0', 0, 4980053911500000256, 999999488, 2, array([ 0.005752,  0.004582], dtype=float32), array([ 0.83245701,  0.57121301], dtype=float32), array([ 0.015625,  0.015625], dtype=float32))
    #time = row[3]/86400.0e9
    #idx = scanlist[i]
    #scan = sc[idx]
    #if scan['startmjd']<=time and scan['endmjd']>=time:
      #swpw[j] = {}
      #swpw[j]['scan'] = scan['scan']
      #swpw[j]['subscan'] = scan['subscan']
      #swpw[j]['antenna'] = int(row[0].strip('Antenna_'))
      #swpw[j]['spw'] = int(row[1].strip('SpectralWindow_'))
      #swpw[j]['time'] = time
      #swpw[j]['pdiff0'] = row[6][0]
      #swpw[j]['pdiff1'] = row[6][1]
      #swpw[j]['psum0'] = row[7][0]
      #swpw[j]['psum1'] = row[7][1]
    ## skip forward until valid time
    #elif scan['startmjd']>time:
      #pass
    ## advance to next scan
    #elif scan['endmjd']<time:
      #i+=1
      ## break if end of scanlist reached
      #if i>=len(scanlist):
	#break
    #else:
      ## This should not happen
      #print "This should not happen!"
      #print row
      #print idx, scan
      #sys.exit()

  ## determine antenna normalization factors from calibrator scan
  normscan = normscan-1
  scan = sc[normscan]
  # get all switched power entries for this particular scan
  sp = {}
  j = 0
  for row in swpw.row:
    time = row[3]/86400.0e9
    if scan['startmjd']<=time and scan['endmjd']>=time:
      sp[j] = {}
      sp[j]['scan'] = scan['scan']
      sp[j]['subscan'] = scan['subscan']
      sp[j]['antenna'] = int(row[0].strip('Antenna_'))
      sp[j]['spw'] = int(row[1].strip('SpectralWindow_'))
      sp[j]['time'] = time
      sp[j]['pdiff0'] = row[6][0]
      sp[j]['pdiff1'] = row[6][1]
      sp[j]['psum0'] = row[7][0]
      sp[j]['psum1'] = row[7][1]
      j+=1
    elif scan['startmjd']>time:
      pass
    else:
      break

  # now we determine the antenna based normalization factors
  norm_factor = n.zeros((28,2))
  for ant in range(0,28,1):
    data0 = []
    data1 = []
    for i in range(len(sp)):
      if sp[i]['antenna'] == ant:
	data0.append(sp[i]['pdiff0'])
	data1.append(sp[i]['pdiff1'])

    if len(data0)!=0:
      norm_factor[ant][0] = n.median(data0)
    else:	
      norm_factor[ant][0] = n.nan
    if len(data1)!=0:
      norm_factor[ant][1] = n.median(data1)
    else:
      norm_factor[ant][1] = n.nan

  # normalize pdiff per antenna to 1 so we can average later
  norm_factor = 1.0/norm_factor

  print "antenna normalization factors for Pdiff"
  print norm_factor

  ## Fill data

  # figure out how many data chunks we expect
  entries = 0
  for scan in scanlist:
    entries += int(n.ceil(sc[scan]['nints']/intchunk*1.0))

  
  pdiff = n.zeros((entries,18,2))
  psum = n.zeros((entries,18,2))

  row = 0
  chunk = 0
  for idx in scanlist:
    nskip = 0
    scan = sc[idx]
    inttime = ((scan['endmjd'] - scan['startmjd'])/scan['nints'])
    
    spwdata_pdiff0 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[]}
    spwdata_pdiff1 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[]}
    spwdata_psum0 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[]}
    spwdata_psum1 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[]}
    
    #  loops through all chunks per scan
    while True:
      starttime = scan['startmjd']+nskip*inttime
      stoptime = scan['startmjd']+(nskip+intchunk)*inttime

      # break if end of scan/subscan reached
      if starttime>scan['endmjd'] or stoptime>scan['endmjd']:
	break

      # loops through all swpw entries matching criteria of a chunk
      while True:
	swpwdata = swpw.row[row]
	time = swpwdata[3]/86400.0e9
	if time<starttime:
	  row+=1
	elif time<=stoptime:
	  spw = int(swpwdata[1].strip('SpectralWindow_'))
	  ant = int(swpwdata[0].strip('Antenna_'))
	  if not n.isnan(norm_factor[ant,0]):
	    spwdata_pdiff0[spw].append(swpwdata[6][0]*norm_factor[ant][0])
	  if not n.isnan(norm_factor[ant,1]):
	    spwdata_pdiff1[spw].append(swpwdata[6][1]*norm_factor[ant][1]) 
	  spwdata_psum0[spw].append(swpwdata[7][0]) 
	  spwdata_psum1[spw].append(swpwdata[7][1])
	  row+=1
	else:
	  # write chunk
	  for i in range(18):
	    if len(spwdata_pdiff0[i])!=0:
	      pdiff[chunk][i][0] = n.mean(spwdata_pdiff0[i])
	    else:
	      pdiff[chunk][i][0] = n.nan
	    if len(spwdata_pdiff1[i])!=0:
	      pdiff[chunk][i][1] = n.mean(spwdata_pdiff1[i])
	    else:
	      pdiff[chunk][i][1] = n.nan
	    if len(spwdata_psum0[i])!=0:
	      psum[chunk][i][0] = n.mean(spwdata_psum0[i])
	    else:
	      psum[chunk][i][0] = n.nan
	    if len(spwdata_psum1[i])!=0:
	      psum[chunk][i][1] = n.mean(spwdata_psum1[i])
	    else:
	      psum[chunk][i][1] = n.nan

	    # reset spwdata for next chunk
	    spwdata_pdiff0[i] = []
	    spwdata_pdiff1[i] = []
	    spwdata_psum0[i] = []
	    spwdata_psum1[i] = []

	  chunk+=1
	  break

      nskip += intchunk

  return n.ma.array(pdiff), n.ma.array(psum)

##################################################
# for plotting
#################################################

def rescale(arr, peak=300):
    return arr * (peak/arr.max())

def rfichan(chan,spw):
    fig = p.figure(figsize=(12,6))
    fig.add_subplot(121)
    x = hadec[:,0]; y = hadec[:,1]
    p.scatter(x, y, s=rescale(rfi[:,chan]), facecolor='k', c='k', alpha=0.1)
    p.xlabel("Local Hour Angle (h)")
    p.ylabel("Dec (deg)")
    fig.add_subplot(122)
    x = altaz[:,1]; y = altaz[:,0]
    p.scatter(x, y, s=rescale(rfi[:,chan]), facecolor='k', c='k', alpha=0.1)
    p.xlabel("Azimuth (deg)")
    p.ylabel("Altitude (deg)")
    print 'Note: circle size is autoscaled.'
    #p.savefig("location_spw%i.png" % i,dpi=300)
    p.show()
    


###########################################################################################

#sdmname = '/lustre/aoc/sciops/fschinze/TSKY0001.sb32701459.eb32715236.57639.510230370375'
#sdmname = '/lustre/aoc/sciops/fschinze/TSKY0001.sb34558063.eb34560858.58029.334773553244'
#scanlist = [3]

pklname = 'rfiarray.pkl'
pklmeta = 'rfimeta.pkl'
pklswpw = 'rfiswpw.pkl'

sdmname = sys.argv[1]
base = sys.argv[1].split('/')[-1]
pklname = 'rfiarray_%s.pkl' % base
pklmeta = 'rfimeta_%s.pkl' % base
pklswpw = 'rfiswpw_%s.pkl' % base
pklswpwsel = 'rfiswpwsel_%s.pkl' % base

print "Reading metadata ..."

if os.path.exists(pklmeta):
   with open(pklmeta, 'r') as pkl:
     sc,sr = pickle.load(pkl)
else:
   sc,sr = read_metadata(sdmname)
   if pklmeta:
     with open(pklmeta,'w') as pkl:
       pickle.dump((sc,sr),pkl)

print "Reading bdfs ..."
if os.path.exists(pklname):
  with open(pklname, 'r') as pkl:
    print "Loading pickle ..."
    rfi, altazstr = pickle.load(pkl)
    scanlist = get_scanlist(sc,"")

else:
  scanlist = get_scanlist(sc,"OBSERVE_TARGET")
  rfi = read_amp(sdmname,scanlist,sc)
  altazstr = calcaltaz(scanlist,sc,sr)
  if pklname:
    with open(pklname,'w') as pkl:
      pickle.dump((rfi,altazstr),pkl)

print "Reading switched power ..."
if os.path.exists(pklswpw):
  with open(pklswpw, 'r') as pkl:
    swpw = pickle.load(pkl)
else:
  scanlist = get_scanlist(sc,"")
  swpw = read_swpw(sdmname)
  if pklswpw:
    with open(pklswpw,'w') as pkl:
      pickle.dump(swpw,pkl)

print "Selecting switched power ..."
if os.path.exists(pklswpwsel):
  with open(pklswpwsel, 'r') as pkl:
    pdiff, psum = pickle.load(pkl)
    scanlist = get_scanlist(sc,"OBSERVE_TARGET")
else:
  scanlist = get_scanlist(sc,"OBSERVE_TARGET")
  pdiff, psum = select_swpw(swpw,sc,scanlist,intchunk=2,normscan=4)
  if pklswpw:
    with open(pklswpwsel,'w') as pkl:
      pickle.dump((pdiff, psum),pkl)

print "finished reading ..."

# distribution of values per scan

# swpw = (swpw_0+swpw_1)/2.0

altaz = calcaltaz(scanlist, sc, sr, format='float')
altazstr = calcaltaz(scanlist, sc, sr, format='str')
radec = calcradec(scanlist, sc, sr)
hadec = n.zeros(altaz.shape,dtype=altaz.dtype)
for i in range(altaz.shape[0]):
         hadec[i][0] = (altaz[i][1]/360.0*24.0)-12.0
         hadec[i][1] = radec[i][1]

spwfreq = get_spwfreq(sdmname,sc)

# plot pointing location
fig = p.figure(figsize=(12,6))
fig.add_subplot(121)
p.plot(hadec[:,0], hadec[:,1], 'k,')
p.xlabel("Local Hour Angle (h)")
p.ylabel("Dec (deg)")
fig.add_subplot(122)
p.plot(altaz[:,1], altaz[:,0], 'k,')
p.xlabel("Azimuth (deg)")
p.ylabel("Altitude (deg)")
# p.show()
p.savefig("pointing_loc_1.png",dpi=300)

# plot pointing location
fig = p.figure(figsize=(12,6))
fig.add_subplot(121)
p.plot(radec[:,0], radec[:,1], 'k,')
p.xlabel("RA (deg)")
p.ylabel("Dec (deg)")
fig.add_subplot(122)
p.plot(altaz[:,1], altaz[:,0], 'k,')
p.xlabel("Azimuth (deg)")
p.ylabel("Altitude (deg)")
# p.show()
p.savefig("pointing_loc_2.png",dpi=300)

fig,ax = plt.subplots(figsize=(16,14))
im = ax.imshow(n.log(rfi), interpolation='nearest', origin='lower', cmap=plt.get_cmap('cubehelix'), aspect='auto')
fig.colorbar(im)
ax.set_xticks(range(0,1024,64))
ax.set_xticklabels(spwfreq)
plt.setp( ax.xaxis.get_majorticklabels(), rotation=80)
ax.set_xlabel('Frequency')
ax.set_yticks(range(0,len(rfi),min(len(rfi), 150)))
ax.set_yticklabels(altazstr[::min(len(rfi), 150)])
ax.set_ylabel('(Alt, Az)')
# p.show()
p.savefig("waterfall.png",dpi=300)


p.figure(figsize=(16,8))
p.subplot(211)
p.plot(n.log(rfi.mean(axis=0)))
p.xlabel('channel')
p.subplot(212)
p.plot(n.log(rfi.mean(axis=1)))
p.xlabel('time sample (averaged)')
# p.show()
p.savefig("time_sample.png",dpi=300)

chperspw = 64
peak = []
for spw in range(16):
    rfich = rfi.mean(axis=0)[spw*chperspw:(spw+1)*chperspw]
    peakch = n.where(rfich == rfich.max())[0][0]
    print '(SPW, Peak channel): (%d, %d)' % (spw, peakch)
    peak.append( (spw, peakch) )

for i in range(16):

	spw, peakch = peak[i]
	rfichan(chperspw*spw + peakch,i)
