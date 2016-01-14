#!/usr/bin/python
import numpy as np
# this code takes excited state information from two files and computes
# properties and observable quantities for evaluation

# open the data file
file1 = 'p3et.energies.dat'
endat = open(file1,"r")
file2 = 'p3et.trdip.dat'
trdat = open(file2,"r")

# define arrays
energy = np.zeros((500,50))
dipole = np.zeros((500,50,5))
stats = np.zeros((50,5))

# read in data
for x in xrange(500):
  templine = endat.readline()
  for y in xrange(50):
    # energies
    templine = endat.readline()
    state,en = templine.split()
    energy[x,y] = float(en)
    # transition dipole moment data
    templine = trdat.readline()
    state,dx,dy,dz,dip,osc = templine.split()
    dipole[x,y,:] = [float(dx),float(dy),float(dz),float(dip),float(osc)]

# basic stats on energy levels
for x in xrange(50):
  stats[x,0] = x
  stats[x,1] = np.nanmin(energy[:,x])
  stats[x,2] = np.nanmax(energy[:,x])
  stats[x,3] = np.average(energy[:,x])
  stats[x,4] = np.std(energy[:,x])
np.savetxt('stats.dat',stats)
    
# compute an absoption spectrum using weighted histogram
# declare constants and arrays
nbins = 50
absorb = np.zeros((nbins))
spec_bin = np.zeros((nbins+1))
spectrum = np.zeros((nbins,2))

# calculate spectrum
absorb,spec_bin = np.histogram(energy,bins=nbins,range=None,normed=True,weights=dipole[:,:,4])
spectrum[:,1] = absorb
for x in xrange(nbins):
  spectrum[x,0] = 0.5*(spec_bin[x]+spec_bin[x+1])
np.savetxt('spectrum.dat',spectrum)

# calculate density of states (unweighted spectrum)
absorb,spec_bin = np.histogram(energy,bins=nbins,range=None,normed=True)
spectrum[:,1] = absorb
for x in xrange(nbins):
  spectrum[x,0] = 0.5*(spec_bin[x]+spec_bin[x+1])
np.savetxt('densofstate.dat',spectrum)

# compute histogram of excited state energies by excited state number
# change constants
nbins = 20
nstates = 10
absorb = np.zeros((nbins))
spec_bin = np.zeros((nbins+1))
spectrum = np.zeros((nbins,2))
# calc spectra
for x in xrange(nstates):
  absorb,spec_bin = np.histogram(energy[:,x],bins=nbins,range=None,normed=True)
  spectrum[:,1] = absorb
  for y in xrange(nbins):
    spectrum[y,0] = 0.5*(spec_bin[y]+spec_bin[y+1])
  s = ""
  file1 = ["exstate",str(x+1),".dat"]
  np.savetxt(s.join(file1),spectrum)
