package ca.bcit.net.algo;

import ca.bcit.net.*;
import ca.bcit.net.demand.Demand;
import ca.bcit.net.demand.DemandAllocationResult;
import ca.bcit.net.modulation.IModulation;
import ca.bcit.net.spectrum.NoSpectrumAvailableException;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Comparator;
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


    private Optional<IModulation> getModulationFromQtable(NetworkNode source, NetworkNode destination, int volume, List<IModulation> modulations, PathPart part) {
        double random = Math.random();

        if (random < EPSILON) {
            // exploit
            INDArray qvalues = getQvalues(source, destination, volume);
            //get the modulation with greatest Q value while fitting the volume (bit rate)
            return modulations.stream()
                    .filter(m -> m.getMaximumDistanceSupportedByBitrateWithJumpsOfTenGbps()[volume] > part.getLength())
                    .max(Comparator.comparingInt(m -> qvalues.getInt(m.getId())));
        } else {
            //explore by random pick
            int pick = (int) Math.floor(Math.random() * modulations.size());
            return Optional.of(modulations.get(pick));
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
                Optional<IModulation> modulationOpt = getModulationFromQtable(source, destination, volume, network.getAllowedModulations(), part);
                if (!modulationOpt.isPresent()){
                    //unable to find a modulation
                    continue pathLoop;
                }
                IModulation modulation = modulationOpt.get();
                double qValue = qTable.getDouble(getNodeId(source), getNodeId(destination), volume, modulation.getId());
                part.setModulation(modulation, (int) Math.round(qValue)); //put Q value as metric

                //TODO get distance, slices, used up spectrum?, used up regenerators?
//                distance += part.getLength();
//                sliceOccupied += part.getSlices().getOccupiedSlices();

                //Original AMRA
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
