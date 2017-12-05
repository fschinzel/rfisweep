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
  for ss in sdm.scans():
    scandict[idx] = {}
    scandict[idx]['source'] = "%s" % ss.source
    scandict[idx]['field'] = "%s" % ss.field
    scandict[idx]['startmjd'] = ss.startMJD
    scandict[idx]['endmjd'] = ss.endMJD
    scandict[idx]['intent'] = ss.intents
    scandict[idx]['scan'] = ss.idx
    scandict[idx]['subscan'] = ss.subidx
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
#  sdm = sdmpy.SDM(sdmname)
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

# Match switched power data to scans
def read_swpw(sdmname,sc,scanlist,intchunk=2):
  import sdmpy.bintab
  s = sdmpy.SDM(sdmname)
  sp = sdmpy.bintab.unpacker(s['SysPower'])
  sp.unpack()

  # create dictionary matched to scan 
  i = 0
  j = 0
  swpw = {}

  # match switched power values to scans and subscans; potentially this step could be skipped 
  # and merged with the step following this one, matching not only to scans/subscans but to
  # integrations
  for row in sp.row:
    time = row[3]/86400.0e9
    idx = scanlist[i]
    scan = sc[idx]
    if scan['startmjd']<=time and scan['endmjd']>=time:
	swpw[j] = {}
	swpw[j]['scan'] = scan['scan']
	swpw[j]['subscan'] = scan['subscan']
	swpw[j]['antenna'] = int(row[0].strip('Antenna_'))
	swpw[j]['spw'] = int(row[1].strip('SpectralWindow_'))
	swpw[j]['power0'] = n.mean(row[7][0])
	swpw[j]['power1'] = n.mean(row[7][1])
	swpw[j]['time'] = time
	j+=1
    elif scan['startmjd']>time:
      pass
    elif scan['endmjd']<time:
      i+=1 
      if i>=len(scanlist):
	break
    else:
      pass

  data_0 = []
  data_1 = []
  for j in range(2):
    if j == 0:
      pol = 'power0'
    else:
      pol = 'power1'
  # need to not only match to sc entry but to intchunk
  # otherwise len(sc) != len(altaz) != len(swpw)
    idx = 0
    data = []
    for i in range(len(scanlist)):
      scan = sc[scanlist[i]]
      inttime = scan['duration']/scan['nints']
      for nskip in range(0, scan['nints']-intchunk+1, intchunk):
	starttime = scan['startmjd']+nskip*inttime
	stoptime = scan['startmjd']+(nskip+intchunk)*inttime
	spwdata = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[]}
      
	# seek start point or skip over scans flagged by scanlist
	while starttime>swpw[idx]['time']:
	  idx+=1
	  if idx>=len(swpw):
	    break

	# fill data
	while swpw[idx]['scan'] == scan['scan'] and swpw[idx]['subscan'] == scan['subscan'] and starttime<=swpw[idx]['time'] and stoptime>=swpw[idx]['time']:
	  spw = swpw[idx]['spw']
	  spwdata[spw].append(swpw[idx][pol])
	  idx+=1
	  if idx>=len(swpw):
	    break

	spws = n.zeros(16)
	for x in range(16):
	  try:
	    spws[x] = n.median(spwdata[x+2],axis=0)
	  except:
	    pass
	data.append(spws)

    data = n.ma.array(data)
    data = n.ma.masked_invalid(data)
    if j == 0:
      data_0 = data
    else:
      data_1 = data

  return data_0, data_1

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
print base
pklname = 'rfiarray_%s.pkl' % base
pklmeta = 'rfimeta_%s.pkl' % base
pklswpw = 'rfiswpw_%s.pkl' % base

print "Reading metadata ..."

print pklmeta
if os.path.exists(pklmeta):
   with open(pklmeta, 'r') as pkl:
     sc,sr = pickle.load(pkl)
else:
   sc,sr = read_metadata(sdmname)
   if pklmeta:
     with open(pklmeta,'w') as pkl:
       pickle.dump((sc,sr),pkl)

scanlist = get_scanlist(sdmname,sc)

print "Scans with less or more than 2 integrations\n"
for i in range(len(sc)):
	if i in scanlist:
		# check bdf filesizes
#		if 24000000.0!=round(os.stat(sc[i]['bdfstr']).st_size,-6):
 #               	print os.stat(sc[i]['bdfstr']).st_size, sc[i]['scan'], sc[i]['subscan']
		# check labeling nints
		if sc[i]['nints'] != 2:
			subscan = int(sc[i]['subscan'])
			row = sc[i-subscan+1]
			bad = sc[i]
			j=1
			while sc[i+j]['scan']==sc[i]['scan']:
				j+=1
			last = sc[i+j-1]
#			print "row start mjd: ", row['startmjd'], "row stop mjd: ", row['endmjd'], "scan: ", row['scan'], "fieldname", bad['source']
			print "nInts: ", bad['nints'], "subscan occurred: ", bad['subscan'], "number of subscans in row: ", last['subscan'], last['intent']
#                        print "subscan start: ", bad['startmjd'],  "subscan stop", bad['endmjd'], "\n"
			if bad['subscan'] == last['subscan']:
				add = "last scan in row"
			else:
				add = "not last scan in row"
			print "%.10f" % bad['startmjd'], "%.10f" % bad['endmjd'], bad['scan'], bad['subscan'], bad['nints'], add
