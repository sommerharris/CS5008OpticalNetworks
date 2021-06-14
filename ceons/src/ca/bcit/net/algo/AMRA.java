package ca.bcit.net.algo;

import ca.bcit.net.*;
import ca.bcit.net.demand.Demand;
import ca.bcit.net.demand.DemandAllocationResult;
import ca.bcit.net.modulation.IModulation;
import ca.bcit.net.spectrum.NoSpectrumAvailableException;

import java.util.List;
import java.util.Optional;

import static ca.bcit.net.Simulation.*;

public class AMRA extends BaseRMSAAlgorithm implements IRMSAAlgorithm {
    public String getKey() {
        return "AMRA";
    }

    ;

    public String getName() {
        return "AMRA";
    }

    ;

    public String getDocumentationURL() {
        return "https://pubsonline.informs.org/doi/pdf/10.1287/opre.24.6.1164";
    }

    ;

    @Override
    public DemandAllocationResult allocateDemand(Demand demand, Network network) {
        try {
            int volume = (int) Math.ceil(demand.getVolume() / 10.0) - 1;
            List<PartedPath> candidatePaths = demand.getCandidatePaths(false, network);

            rankCandidatePaths(network, volume, candidatePaths);

            allocateWorkingPath(demand, candidatePaths);

            if (shouldAllocateBackupPath(demand, candidatePaths)) {
                int backupVolume = (int) Math.ceil(demand.getSqueezedVolume() / 10.0) - 1;

                rankCandidatePaths(network, backupVolume, demand.getCandidatePaths(true, network));

                allocateBackupPath(demand, candidatePaths);
            }

            return new DemandAllocationResult(demand);
        } catch (NoSpectrumAvailableException e) {
            return DemandAllocationResult.NO_SPECTRUM;
        } catch (NoRegeneratorsAvailableException | NetworkException e) {
            return DemandAllocationResult.NO_REGENERATORS;
        }
    }


    private int getModulationId(NetworkNode source, NetworkNode destination, int volume, List<IModulation> modulations) {
        double random = Math.random();
//        NetworkNode source = pathPart.getSource();
//        NetworkNode destination = pathPart.getDestination();
       

        if (random < EPSILON) {
            //get the modulation id with greatest Q value using argMax method.
            return getQvalues(source, destination, volume).argMax().getInt(0);
//            return modulations.stream().filter(m -> m.getId() == modulationId).findAny();

        } else {
            //random pick
            int floor = (int) Math.floor(Math.random() * modulations.size());
//            return Optional.of(modulations.get(floor));
            return modulations.get(floor).getId();
        }

    }

    protected void applyMetricsToCandidatePaths(Network network, int volume, List<PartedPath> candidatePaths) {
        pathLoop:
        for (PartedPath path : candidatePaths) {
            path.mergeRegeneratorlessParts();

            long distance = 0;
            long sliceOccupied = 0;
        
            // choosing modulations for parts
            for (PathPart part : path) {
                NetworkNode source = part.getSource();
                NetworkNode destination = part.getDestination();
                
                //select modulation from Q table
                List<IModulation> allowedModulations = network.getAllowedModulations();
                int modulationId = getModulationId(source, destination, volume, allowedModulations);
                double qValue = qTable.getDouble(getNodeId(source), getNodeId(destination), volume, modulationId);
                
                if (modulationId < 0 || modulationId > 5) { //unable to find a modulation
                    continue pathLoop;
                }
                Optional<IModulation> modulationOpt =  allowedModulations.stream().filter(m -> m.getId() == modulationId).findAny();
                if (!modulationOpt.isPresent()){
                    continue  pathLoop;
                }
                IModulation modulation = modulationOpt.get();
                if (modulation.getMaximumDistanceSupportedByBitrateWithJumpsOfTenGbps()[volume] >= part.getLength()) {
                    part.setModulation(modulation, (int) Math.round(qValue));
//                    part.setModulationIfBetter(modulation, calculateModulationMetric(network, part, modulation));
                } else {
                    //TODO should change to choose the available modulation with max q value
                    continue pathLoop;
                }

                //TODO get distance, slices, used up spectrum?, used up regenerators?
//                distance += part.getLength();
//                sliceOccupied += part.getSlices().getOccupiedSlices();

                //Original ARMA
//				for (IModulation modulation : network.getAllowedModulations())
//					if (modulation.getMaximumDistanceSupportedByBitrateWithJumpsOfTenGbps()[volume] >= part.getLength())
//						part.setModulationIfBetter(modulation, calculateModulationMetric(network, part, modulation));

				if (part.getModulation() == null)
					continue pathLoop;
            }
            path.calculateMetricFromParts();
            path.mergeIdenticalModulation(volume);

            // Unify modulations if needed
            if (!network.canSwitchModulation()) {
                IModulation modulation = path.getModulationFromLongestPart();
                for (PathPart part : path)
                    part.setModulation(modulation, calculateModulationMetric(network, part, modulation));
                path.calculateMetricFromParts();
            }

            // Update metrics
            // TODO calculate reward for each part and sum all rewards for a candidate path

//			int increment = - reward;
            int increment = network.getRegeneratorMetricValue() * path.getNeededRegeneratorsCount();
            path.setMetric(path.getMetric() + increment);
        }
    }

    protected void filterCandidatePaths(List<PartedPath> candidatePaths) {
        for (int i = 0; i < candidatePaths.size(); i++)
            if (candidatePaths.get(i).getMetric() < 0) {
                candidatePaths.remove(i);
                i--;
            }
    }

    private static int calculateModulationMetric(Network network, PathPart part, IModulation modulation) {
        double slicesOccupationPercentage = part.getOccupiedSlicesPercentage() * 100;

        return network.getDynamicModulationMetric(modulation, getSlicesOccupationMetric(slicesOccupationPercentage));
    }

    private static int getSlicesOccupationMetric(double slicesOccupationPercentage) {
        if (slicesOccupationPercentage > 90)
            return 5;
        else if (slicesOccupationPercentage > 75)
            return 4;
        else if (slicesOccupationPercentage > 60)
            return 3;
        else if (slicesOccupationPercentage > 40)
            return 2;
        else if (slicesOccupationPercentage > 20)
            return 1;

        return 0;
    }
}
