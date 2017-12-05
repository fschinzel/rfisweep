#!/usr/bin/env python2.7
# -*- coding: iso-8859-1 -*-

# Extract data from SDM for plotting RFI
# This script was inspired by Casey Law who performed a similar analysis for VLASS a couple 
# years ago, some of his code was reused here. However it is now compatible to only require 
# sdmpy to interacting with the SDM/BDF, which is mainted by Paul Demorest.
# -- Frank Schinzel (10/06/2017)

import sdmpy
import ephem
import pickle
import os
from contextlib import closing
import multiprocessing as mp
import numpy as n
import sys
import math

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pylab as p

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

def get_scanlist(sdmname,sc):
  sdm = sdmpy.SDM(sdmname)
  scanlist = []
  for idx in range(len(sc)):
    if "OBSERVE_TARGET" in sc[idx]['intent']:
      scanlist.append(idx)
#  last_scan = sc.keys()[-1]
#  scanlist = range(1,last_scan,1)
  return scanlist

def calcaltaz(scanlist, sc, sr, format='str', intchunk=2):
    """ Calculates a single (alt,az) per scan in scanlist.
    """

    inttime = (sc[0]['endmjd'] - sc[1]['startmjd'])*24*3600/sc[1]['nints']

    vla = ephem.Observer()
    vla.lat = '34:04:43.497'
    vla.long = '-107:37:03.819'
    vla.elevation = 2124
    src = ephem.FixedBody()


    altaz = []
    for scan in scanlist:
        src._ra, src._dec = [(sr[srn]['ra'], sr[srn]['dec']) for srn in sr.keys() if sc[scan]['source'] == sr[srn]['source']][0]
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
    p.savefig("location_spw%i.png" % i,dpi=300)
    #p.show()
    


###########################################################################################

#sdmname = '/lustre/aoc/sciops/fschinze/TSKY0001.sb32701459.eb32715236.57639.510230370375'
#scanlist = [3]

#pklname = 'rfiarray.pkl'
#pklmeta = 'rfimeta.pkl'

sdmname = sys.argv[1]
base = sys.argv[1].split('/')[-1]
pklname = 'rfiarray_%s.pkl' % base
pklmeta = 'rfimeta_%s.pkl' % base

print "Reading metadata ..."

if os.path.exists(pklmeta):
  with open(pklmeta, 'r') as pkl:
    print "Loading pickle"
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
    scanlist = get_scanlist(sdmname,sc)

else:
  scanlist = get_scanlist(sdmname,sc)
  rfi = read_amp(sdmname,scanlist,sc)
  altazstr = calcaltaz(scanlist,sc,sr)
  if pklname:
    with open(pklname,'w') as pkl:
      pickle.dump((rfi,altazstr),pkl)

altaz = calcaltaz(scanlist, sc, sr, format='float')
radec = calcradec(scanlist, sc, sr)
hadec = n.zeros(altaz.shape,dtype=altaz.dtype)
for i in range(altaz.shape[0]):
        hadec[i][0] = (altaz[i][1]/360.0*24.0)-12.0
        hadec[i][1] = radec[i][1]

spwfreq = get_spwfreq(sdmname,sc)

# plotting
print "Plotting ..."

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
#p.show()
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
#p.show()
p.savefig("waterfall.png",dpi=300)


p.figure(figsize=(16,8))
p.subplot(211)
p.plot(n.log(rfi.mean(axis=0)))
p.xlabel('channel')
p.subplot(212)
p.plot(n.log(rfi.mean(axis=1)))
p.xlabel('time sample (averaged)')
#p.show()
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
