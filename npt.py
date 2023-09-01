#!/home/emmanuel/mambaforge/bin/python

##########################################################################
#
# Simulated the NPT ensemble
#
from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from sys import stdout
import numpy as np
import sys

irun = sys.argv[1]
sequence = sys.argv[2]
prev = str(int(irun) - 1)


temp = 300
timestep = 2.0
cutoff = 1.2
printfreq = 5000
nsteps = 10000000  # 20 ns run
nsav_rst = 1000000  # Save every nanosecond
gamma = 5

print('Checking for CUDA...')
platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}

prmtop = AmberPrmtopFile('dna-%s.prmtop' % sequence)
inpcrd = AmberInpcrdFile('noe-%s.rst' % sequence)

print ('Setting up system...')
system = prmtop.createSystem(nonbondedMethod=PME,
    nonbondedCutoff=cutoff*unit.nanometers, constraints=HBonds, rigidWater=True,
    ewaldErrorTolerance=0.00005)

print('Choosing integrator...')
integrator = LangevinIntegrator(temp*unit.kelvin, gamma/unit.picoseconds,
                                timestep*unit.femtoseconds)
integrator.setConstraintTolerance(0.000001)
system.addForce(MonteCarloBarostat(1*unit.atmospheres, temp*unit.kelvin, 25))

print('Establishing simulation context...')
simulation = Simulation(prmtop.topology, system, integrator)#, platform, properties)

if irun == '1' and inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setVelocities(inpcrd.velocities)
else:
    with open(prev+"-"+sequence+'.chk', 'rb') as f: simulation.context.loadCheckpoint(f.read())

simulation.reporters.append(CheckpointReporter(irun+"-"+sequence+'.chk', nsav_rst))
simulation.reporters.append(DCDReporter(irun+"-"+sequence+'.dcd', printfreq))
simulation.reporters.append(StateDataReporter(stdout, printfreq, step=True,
                                              time=True, potentialEnergy=True,
                                              kineticEnergy=True, totalEnergy=True,
                                              temperature=True, density=True, speed=True,
                                              totalSteps=printfreq, separator='\t'))

simulation.step(nsteps)
