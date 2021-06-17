package ca.bcit.net.algo;

import ca.bcit.net.*;
import ca.bcit.net.demand.Demand;
import ca.bcit.net.demand.DemandAllocationResult;
import ca.bcit.net.spectrum.NoSpectrumAvailableException;
import ca.bcit.net.spectrum.Spectrum;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.OptionalDouble;

import static ca.bcit.net.Simulation.EPSILON;

public abstract class BaseRMSAAlgorithm {
    public DemandAllocationResult allocateDemand(Demand demand, Network network) throws InstantiationException, ClassNotFoundException, IllegalAccessException {
        try {
            int volume = (int) Math.ceil(demand.getVolume() / 10.0) - 1;
            List<PartedPath> candidatePaths = demand.getCandidatePaths(false, network);

            rankCandidatePaths(network, volume, candidatePaths);

            allocateWorkingPath(demand, candidatePaths);

            if (shouldAllocateBackupPath(demand, candidatePaths))
                allocateBackupPath(demand, candidatePaths);

            return new DemandAllocationResult(demand);
        } catch (NoSpectrumAvailableException e) {
            return DemandAllocationResult.NO_SPECTRUM;
        } catch (NoRegeneratorsAvailableException | NetworkException e) {
            return DemandAllocationResult.NO_REGENERATORS;
        }
    }

    protected void rankCandidatePaths(Network network, int volume, List<PartedPath> candidatePaths) {
        if (candidatePaths.isEmpty())
            throw new NoSpectrumAvailableException("There are no candidate paths to allocate the demand.");

        applyMetricsToCandidatePaths(network, volume, candidatePaths);

        //No need to filter
//		filterCandidatePaths(candidatePaths);

        if (candidatePaths.isEmpty())
            throw new NoRegeneratorsAvailableException("There are no candidate paths to allocate the demand.");

        sortCandidatePaths(candidatePaths);
    }

    protected boolean allocateToHighestRankedPathAvailable(Demand demand, List<PartedPath> candidatePaths) throws NetworkException {
        double random = 0;//Math.random();
        if (random < EPSILON) {
            //exploit - allocate according to metric - candidatePaths are sorted by metric in ascending order
            for (PartedPath path : candidatePaths) {
                //try to allocate
                boolean allocateResult = tryAllocate(demand, path);
                if (allocateResult) {
                    updateQtable(demand, path, true);
                    return true;
                }
            }
        } else {
            if (candidatePaths.isEmpty()) {
                return false;
            }
            List<PartedPath> paths = new ArrayList<>(candidatePaths);
            //explore by random pick
            while (!paths.isEmpty()) {
                int pick = (int) Math.floor(Math.random() * paths.size());
                PartedPath path = candidatePaths.get(pick);

                boolean allocateResult = tryAllocate(demand, path);
                if (allocateResult) {
                    updateQtable(demand, path, true);
                    return true;
                }
                paths.remove(pick);
            }
        }
        //original
//		for (PartedPath path : candidatePaths)
//			if (demand.allocate(path))
//				return true;
        for (PartedPath path : candidatePaths) {
            updateQtable(demand, path, false);
        }
        return false;
    }

    private void updateQtable(Demand demand, PartedPath path, boolean allocateResult) {
        //calculate reward
        int reward = allocateResult ? 100 : -1000;

        //Update Q table
        for (PathPart part : path.getParts()) {
            part.getSlices();
            double occupiedSlicesPercentage = part.getOccupiedSlicesPercentage();
            reward *= (0.5-occupiedSlicesPercentage);
            Simulation.updateQtable(reward, demand.getVolume(), part);
        }
    }

    private boolean tryAllocate(Demand demand, PartedPath path) {
        boolean allocateResult;
        try {
            allocateResult = demand.allocate(path);
        } catch (NetworkException e) {
            allocateResult = false;
        }

//        updateQtable(demand, path, allocateResult);

        return allocateResult;
    }

    protected void sortCandidatePaths(List<PartedPath> candidatePaths) {
        Collections.sort(candidatePaths);
    }

    protected void filterCandidatePaths(List<PartedPath> candidatePaths) {
    }

    protected boolean shouldAllocateBackupPath(Demand demand, List<PartedPath> candidatePaths) {
        return demand.allocateBackup() && !candidatePaths.isEmpty();
    }

    protected void allocateWorkingPath(Demand demand, List<PartedPath> candidatePaths) {
        boolean allocated = allocateToHighestRankedPathAvailable(demand, candidatePaths);
//		if (!allocated){
//			System.out.println("Working path not allocated ");
//		}
    }

    protected void allocateBackupPath(Demand demand, List<PartedPath> candidatePaths) {
        allocateToHighestRankedPathAvailable(demand, candidatePaths);
    }

    protected abstract void applyMetricsToCandidatePaths(Network network, int volume, List<PartedPath> candidatePaths);
}
