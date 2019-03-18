import simtk.openmm as mm
import simtk.unit as unit
from simtk.openmm.app import Topology, PDBFile, Modeller, ForceField, PDBxFile, PME, Simulation, StateDataReporter
import numpy as np
from collections import namedtuple
from functools import reduce
import os
import re

class Metadynamics(object):
    """Performs metadynamics.

    This class implements well-tempered metadynamics, as described in Barducci et al.,
    "Well-Tempered Metadynamics: A Smoothly Converging and Tunable Free-Energy Method"
    (https://doi.org/10.1103/PhysRevLett.100.020603).  You specify from one to three
    collective variables whose sampling should be accelerated.  A biasing force that
    depends on the collective variables is added to the simulation.  Initially the bias
    is zero.  As the simulation runs, Gaussian bumps are periodically added to the bias
    at the current location of the simulation.  This pushes the simulation away from areas
    it has already explored, encouraging it to sample other regions.  At the end of the
    simulation, the bias function can be used to calculate the system's free energy as a
    function of the collective variables.

    To use the class you create a Metadynamics object, passing to it the System you want
    to simulate and a list of BiasVariable objects defining the collective variables.
    It creates a biasing force and adds it to the System.  You then run the simulation
    as usual, but call step() on the Metadynamics object instead of on the Simulation.

    You can optionally specify a directory on disk where the current bias function should
    periodically be written.  In addition, it loads biases from any other files in the
    same directory and includes them in the simulation.  It loads files when the
    Metqdynamics object is first created, and also checks for any new files every time it
    updates its own bias on disk.

    This serves two important functions.  First, it lets you stop a metadynamics run and
    resume it later.  When you begin the new simulation, it will load the biases computed
    in the earlier simulation and continue adding to them.  Second, it provides an easy
    way to parallelize metadynamics sampling across many computers.  Just point all of
    them to a shared directory on disk.  Each process will save its biases to that
    directory, and also load in and apply the biases added by other processes.
    """

    def __init__(self, system, variables, temperature, biasFactor, height, frequency, saveFrequency=None, biasDir=None):
        """Create a Metadynamics object.

        Parameters
        ----------
        system: System
            the System to simulate.  A CustomCVForce implementing the bias is created and
            added to the System.
        variables: list of BiasVariables
            the collective variables to sample
        temperature: temperature
            the temperature at which the simulation is being run.  This is used in computing
            the free energy.
        biasFactor: float
            used in scaling the height of the Gaussians added to the bias.  The collective
            variables are sampled as if the effective temperature of the simulation were
            temperature*biasFactor.
        height: energy
            the initial height of the Gaussians to add
        frequency: int
            the interval in time steps at which Gaussians should be added to the bias potential
        saveFrequency: int (optional)
            the interval in time steps at which to write out the current biases to disk.  At
            the same it it writes biases, it also checks for updated biases written by other
            processes and loads them in.  This must be a multiple of frequency.
        biasDir: str (optional)
            the directory to which biases should be written, and from which biases written by
            other processes should be loaded
        """
        if not unit.is_quantity(temperature):
            temperature = temperature*unit.kelvin
        if not unit.is_quantity(height):
            height = height*unit.kilojoules_per_mole
        if biasFactor < 1.0:
            raise ValueError('biasFactor must be >= 1')
        if (saveFrequency is None and biasDir is not None) or (saveFrequency is not None and biasDir is None):
            raise ValueError('Must specify both saveFrequency and biasDir')
        if saveFrequency is not None and (saveFrequency < frequency or saveFrequency%frequency != 0):
            raise ValueError('saveFrequency must be a multiple of frequency')
        self.variables = variables
        self.temperature = temperature
        self.biasFactor = biasFactor
        self.height = height
        self.frequency = frequency
        self.biasDir = biasDir
        self.saveFrequency = saveFrequency
        self._id = np.random.randint(0xFFFFFFFFFFFF)
        self._saveIndex = 0
        self._selfBias = np.zeros(tuple(v.gridWidth for v in variables))
        self._totalBias = np.zeros(tuple(v.gridWidth for v in variables))
        self._loadedBiases = {}
        self._deltaT = temperature*(biasFactor-1)
        varNames = ['cv%d' % i for i in range(len(variables))]
 
        self._force = mm.CustomCVForce('table(%s)' % ', '.join(varNames))
        for name, var in zip(varNames, variables):
            self._force.addCollectiveVariable(name, var.force)

        widths = [v.gridWidth for v in variables]
        mins = [v.minValue for v in variables]
        maxs = [v.maxValue for v in variables]
        if len(variables) == 1:
            self._table = mm.Continuous1DFunction(self._totalBias.flatten(), mins[0], maxs[0]) # stores BiasInfo, min and max values
        elif len(variables) == 2:
            self._table = mm.Continuous2DFunction(widths[0], widths[1], self._totalBias.flatten(), mins[0], maxs[0], mins[1], maxs[1])
        elif len(variables) == 3:
            self._table = mm.Continuous3DFunction(widths[0], widths[1], widths[2], self._totalBias.flatten(), mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
        else:
            raise ValueError('Metadynamics requires 1, 2, or 3 collective variables')
        self._force.addTabulatedFunction('table', self._table)
        self._force.setForceGroup(1)
        system.addForce(self._force)
        self._syncWithDisk()

    def step(self, simulation, steps):
        """Advance the simulation by integrating a specified number of time steps.

        Parameters
        ----------
        simulation: Simulation
            the Simulation to advance
        steps: int
            the number of time steps to integrate
        """
        stepsToGo = steps
        # JG: reset the simulation steps after minimization
        simulation.currentStep = 0
        while stepsToGo > 0:
            nextSteps = stepsToGo
            if simulation.currentStep % self.frequency == 0:
                nextSteps = min(nextSteps, self.frequency)
            else:
                nextSteps = min(nextSteps, simulation.currentStep % self.frequency)
            simulation.step(nextSteps)

            if simulation.currentStep % self.frequency == 0:
                # peastman original code:
                position = self._force.getCollectiveVariableValues(simulation.context)
                print("Gaussian position:")
                print(position)

                # peastman original code (refers to an old version where "groups" is a dict):
                #energy = simulation.context.getState(getEnergy=True, groups={31}).getPotentialEnergy()

                # JG: the “groups” parameter in getState() accepts 32-bit int (i.e. 2**n for group n)
                energy = simulation.context.getState(getEnergy=True, groups=2).getPotentialEnergy()
                print("CustomCVForce potential energy:")
                print(energy)
                height = self.height*np.exp(-energy/(unit.MOLAR_GAS_CONSTANT_R*self._deltaT))
                # JG: output the bias values added each step
                print('Bias height:',height)
                self._addGaussian(position, height, simulation.context)

            if self.saveFrequency is not None and simulation.currentStep % self.saveFrequency == 0:
                self._syncWithDisk()
            stepsToGo -= nextSteps

    # JG: this function later called to plot the free energy surface
    def getFreeEnergy(self):
        """Get the free energy of the system as a function of the collective variables.

        The result is returned as a N-dimensional NumPy array, where N is the number of collective
        variables.  The values are in kJ/mole.  The i'th position along an axis corresponds to
        minValue + i*(maxValue-minValue)/gridWidth.
        """
        return -((self.temperature+self._deltaT)/self._deltaT)*self._totalBias

    # JG: this function was later called to plot collective variables
    def getCollectiveVariables(self, simulation):
        """Get the current values of all collective variables in a Simulation."""
        return self._force.getCollectiveVariableValues(simulation.context)

    def _addGaussian(self, position, height, context):
        """Add a Gaussian to the bias function."""
        # Compute a Gaussian along each axis.

        axisGaussians = []
        for i,v in enumerate(self.variables):
            x = (position[i]-v.minValue) / (v.maxValue-v.minValue)
            if v.periodic:
                x = x % 1.0
            # peastman original code: assuming len(dist) == v.gridWidth but they are not always equal due to numerical instability):
            #dist = np.abs(np.arange(0, 1, 1.0/v.gridWidth) - x) 

            # JG: use linspace (better stability) instead so len(dist) and v.gridWidth are always equal                
            dist = np.abs(np.linspace(0, 1.0, num = v.gridWidth) - x)  
 
            if v.periodic:
                dist = np.min(np.array([dist, np.abs(dist-1)]), axis=0)
            axisGaussians.append(np.exp(-dist*dist*v.gridWidth/v.biasWidth))

        # Compute their outer product.
        if len(self.variables) == 1:
            gaussian = axisGaussians[0]
        else:
            gaussian = reduce(np.multiply.outer, reversed(axisGaussians))

        # Add it to the bias.

        height = height.value_in_unit(unit.kilojoules_per_mole)
        self._selfBias += height*gaussian
        self._totalBias += height*gaussian
        widths = [v.gridWidth for v in self.variables]
        mins = [v.minValue for v in self.variables]
        maxs = [v.maxValue for v in self.variables]
        if len(self.variables) == 1:
            self._table.setFunctionParameters(self._totalBias.flatten(), mins[0], maxs[0])
        elif len(self.variables) == 2:
            self._table.setFunctionParameters(widths[0], widths[1], self._totalBias.flatten(), mins[0], maxs[0], mins[1], maxs[1])
        elif len(self.variables) == 3:
            self._table.setFunctionParameters(widths[0], widths[1], widths[2], self._totalBias.flatten(), mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])

        self._force.updateParametersInContext(context)

    def _syncWithDisk(self):
        """Save biases to disk, and check for updated files created by other processes."""
        if self.biasDir is None:
            return

        # Use a safe save to write out the biases to disk, then delete the older file.

        oldName = os.path.join(self.biasDir, 'bias_%d_%d.npy' % (self._id, self._saveIndex))
        self._saveIndex += 1
        tempName = os.path.join(self.biasDir, 'temp_%d_%d.npy' % (self._id, self._saveIndex))
        fileName = os.path.join(self.biasDir, 'bias_%d_%d.npy' % (self._id, self._saveIndex))
        np.save(tempName, self._selfBias)
        os.rename(tempName, fileName)
        if os.path.exists(oldName):
            os.remove(oldName)

        # Check for any files updated by other processes.

        fileLoaded = False
        pattern = re.compile('bias_(.*)_(.*)\.npy')
        for filename in os.listdir(self.biasDir):
            match = pattern.match(filename)
            if match is not None:
                matchId = int(match.group(1))
                matchIndex = int(match.group(2))
                if matchId != self._id and (matchId not in self._loadedBiases or matchIndex > self._loadedBiases[matchId].index):
                    data = np.load(os.path.join(self.biasDir, filename))
                    self._loadedBiases[matchId] = _LoadedBias(matchId, matchIndex, data)
                    fileLoaded = True

        # If we loaded any files, recompute the total bias from all processes.

        if fileLoaded:
            self._totalBias = self._selfBias
            for bias in self._loadedBiases.values():
                self._totalBias += bias.bias


class BiasVariable(object):
    """A collective variable that can be used to bias a simulation with metadynamics."""

    def __init__(self, force, minValue, maxValue, biasWidth, periodic=False, gridWidth=None):
        """Create a BiasVariable.

        Parameters
        ----------
        force: Force
            the Force object whose potential energy defines the collective variable
        minValue: float
            the minimum value the collective variable can take.  If it should ever go below this,
            the bias force will be set to 0.
        maxValue: float
            the maximum value the collective variable can take.  If it should ever go above this,
            the bias force will be set to 0.
        biasWidth: float
            the width (standard deviation) of the Gaussians added to the bias during metadynamics
        periodic: bool
            whether this is a periodic variable, such that minValue and maxValue are physical equivalent
        gridWidth: int
            the number of grid points to use when tabulating the bias function.  If this is omitted,
            a reasonable value is chosen automatically.
        """
        self.force = force
        self.minValue = minValue
        self.maxValue = maxValue
        self.biasWidth = biasWidth
        self.periodic = periodic
        if gridWidth is None:
            self.gridWidth = int(np.ceil(5*(maxValue-minValue)/biasWidth))
        else:
            self.gridWidth = gridWidth


_LoadedBias = namedtuple('LoadedBias', ['id', 'index', 'bias'])
