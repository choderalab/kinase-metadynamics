import logging
import os
from pdbfixer import PDBFixer
import simtk.openmm as mm
from simtk.openmm import unit 
from simtk.openmm.app import Topology, PDBFile, Modeller, ForceField, PDBxFile, PME, Simulation, StateDataReporter, DCDReporter
import protein_features as pf
import metadynamics as mtd
import matplotlib.pyplot as plot 
import numpy as np

## Setup general logging (guarantee output/error message in case of interruption)
logger = logging.getLogger(__name__)
logging.root.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)
'''
## clean up the input pdb file using pdbfixer and load using Modeller

# fix using pdbfixer: remove the ligand but keep the crystal waters 
fixer = PDBFixer(pdbid='5UG9')
fixer.findMissingResidues()

# modify missingResidues so the extra residues on the end are ignored
#fixer.missingResidues = {(0,47): fixer.missingResidues[(0,47)]}
fixer.missingResidues = {}

# remove ligand but keep crystal waters
fixer.removeHeterogens(True)
print("Done removing heterogens.")

# find missing atoms/terminals
fixer.findMissingAtoms()
if fixer.missingAtoms or fixer.missingTerminals:
    fixer.addMissingAtoms()
    print("Done adding atoms/terminals.")
else:
    print("No atom/terminal needs to be added.")

# add hydrogens
fixer.addMissingHydrogens(7.0)
print("Done adding hydrogens.")
# output fixed pdb
PDBFile.writeFile(fixer.topology, fixer.positions, open('test_fixed.pdb', 'w'), keepIds=True)
print("Done outputing the fixed pdb file.")
'''

# load pdb to Modeller
pdb = PDBFile('5UG9_fixed.pdb')
molecule = Modeller(pdb.topology,pdb.positions)
print("Done loading pdb to Modeller.")
# load force field
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
print("Done loading force field.")
# prepare system
molecule.addSolvent(forcefield, padding=12*unit.angstrom, model='tip3p', positiveIon='Na+', negativeIon='Cl-', ionicStrength=0*unit.molar)
print("Done adding solvent.")
PDBxFile.writeFile(molecule.topology,molecule.positions,open("5UG9_fixed.pdbx", 'w'))
print("Done outputing pdbx.")
system = forcefield.createSystem(molecule.topology, nonbondedMethod=PME, rigidWater=True, nonbondedCutoff=1*unit.nanometer)

# add the custom cv force
# Specify the set of key atoms and calculate key dihedrals and distances
(dih, dis) = pf.main('5UG9','A')
# dihedrals
dih_0 = mm.CustomTorsionForce("theta")
dih_0.addTorsion(int(dih[0][0]), int(dih[0][1]), int(dih[0][2]), int(dih[0][3]))
dih_1 = mm.CustomTorsionForce("theta")
dih_1.addTorsion(int(dih[1][0]), int(dih[1][1]), int(dih[1][2]), int(dih[1][3]))
dih_2 = mm.CustomTorsionForce("theta")
dih_2.addTorsion(int(dih[2][0]), int(dih[2][1]), int(dih[2][2]), int(dih[2][3]))
dih_3 = mm.CustomTorsionForce("theta")
dih_3.addTorsion(int(dih[3][0]), int(dih[3][1]), int(dih[3][2]), int(dih[3][3]))
dih_4 = mm.CustomTorsionForce("theta")
dih_4.addTorsion(int(dih[4][0]), int(dih[4][1]), int(dih[4][2]), int(dih[4][3]))
dih_5 = mm.CustomTorsionForce("theta")
dih_5.addTorsion(int(dih[5][0]), int(dih[5][1]), int(dih[5][2]), int(dih[5][3]))
dih_6 = mm.CustomTorsionForce("theta")
dih_6.addTorsion(int(dih[6][0]), int(dih[6][1]), int(dih[6][2]), int(dih[6][3]))
dih_7 = mm.CustomTorsionForce("theta")
dih_7.addTorsion(int(dih[7][0]), int(dih[7][1]), int(dih[7][2]), int(dih[7][3]))
# distances
dis_0 = mm.CustomBondForce("r")
dis_0.addBond(int(dis[0][0]), int(dis[0][1]))
dis_1 = mm.CustomBondForce("r")
dis_1.addBond(int(dis[1][0]), int(dis[1][1]))
dis_2 = mm.CustomBondForce("r")
dis_2.addBond(int(dis[2][0]), int(dis[2][1]))
dis_3 = mm.CustomBondForce("r")
dis_3.addBond(int(dis[3][0]), int(dis[3][1]))
dis_4 = mm.CustomBondForce("r")
dis_4.addBond(int(dis[4][0]), int(dis[4][1]))
print("Done populating dihedrals and distances.")
# Specify a unique CustomCVForce
# the trial CV = dih1 + dih2 + ... + dih8 + dis1 + ... + dis5
cv_force = mm.CustomCVForce("dih_0 + dih_1 + dih_2 + dih_3 + dih_4 + dih_5 + dih_6 + dih_7 + dis_0 + dis_1 + dis_2 + dis_3 + dis_4")
cv_force.addCollectiveVariable('dih_0', dih_0)
cv_force.addCollectiveVariable('dih_1', dih_1)
cv_force.addCollectiveVariable('dih_2', dih_2)
cv_force.addCollectiveVariable('dih_3', dih_3)
cv_force.addCollectiveVariable('dih_4', dih_4)
cv_force.addCollectiveVariable('dih_5', dih_5)
cv_force.addCollectiveVariable('dih_6', dih_6)
cv_force.addCollectiveVariable('dih_7', dih_7)
cv_force.addCollectiveVariable('dis_0', dis_0)
cv_force.addCollectiveVariable('dis_1', dis_1)
cv_force.addCollectiveVariable('dis_2', dis_2)
cv_force.addCollectiveVariable('dis_3', dis_3)
cv_force.addCollectiveVariable('dis_4', dis_4)
bv = mtd.BiasVariable(cv_force, -np.pi*8, np.pi*8 + 10*5, 0.5, True)
print("Done adding forces.")

# specify the rest of the context for minimization
integrator = mm.VerletIntegrator(0.5*unit.femtoseconds)
print("Done specifying integrator.")
platform = mm.Platform.getPlatformByName('CUDA')
print("Done specifying platform.")
simulation = Simulation(molecule.topology, system, integrator, platform)
print("Done specifying simulation.")
simulation.context.setPositions(molecule.positions)
print("Done recording a context for positions.")
simulation.context.setVelocitiesToTemperature(310.15*unit.kelvin)
print("Done assigning velocities.")

# start minimization
tolerance = 0.1*unit.kilojoules_per_mole/unit.angstroms
print("Done setting tolerance.")
simulation.minimizeEnergy(tolerance=tolerance,maxIterations=1000)
print("Done setting energy minimization.")
simulation.reporters.append(StateDataReporter('relax-hydrogens.log', 1000, step=True, temperature=True, potentialEnergy=True, totalEnergy=True, speed=True))
simulation.step(10000)
print("Done 10000 steps of simulation.")
positions = simulation.context.getState(getPositions=True).getPositions()
print("Done updating positions.")
simulation.saveCheckpoint('state.chk')
print("Done saving checkpoints.")

# add customCVForce as force group 1 to system
cv_force.setForceGroup(1)
system.addForce(cv_force)

# update the current context with changes in system
simulation.context.reinitialize()

# Set up the context for mtd simulation
# at this step the CV and the system are separately passed to Metadynamics
meta = mtd.Metadynamics(system, [bv], 310.0*unit.kelvin, 1000./310., 1.2*unit.kilojoules_per_mole, 500, saveFrequency=500, biasDir='./biases')
integrator = mm.LangevinIntegrator(310*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
print("Done specifying integrator.")
simulation = Simulation(molecule.topology, system, integrator)
print("Done specifying simulation.")
simulation.context.setPositions(positions)
print("Done setting up the simulation.")

# equilibration
simulation.context.setVelocitiesToTemperature(310*unit.kelvin)
simulation.step(100)
print("Done 100 steps of equilibration.")

# set simulation reporters
simulation.reporters.append(DCDReporter('mtd_5UG9.dcd', 1000))
simulation.reporters.append(StateDataReporter('mtd_5UG9.out', 10000, step=True, 
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=5000000, separator='\t'))

# Run the simulation (10 ns, 5*10^6 steps) and plot the free energy landscape
meta.step(simulation, 5000000)
#plot.imshow(meta.getFreeEnergy())
#plot.show()
print("Done with 10 ns (5E+6 steps) of production run.")
