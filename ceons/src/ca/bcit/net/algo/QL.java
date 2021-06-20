package ca.bcit.net.algo;

import ca.bcit.Tuple;
import ca.bcit.net.*;
import ca.bcit.net.demand.Demand;
import ca.bcit.net.demand.DemandAllocationResult;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.stream.Collectors;

import static ca.bcit.net.Simulation.*;

public class QL implements IRMSAAlgorithm{

	public static final int NEG_REWARD = -3500;

	public String getKey(){
		return "QL";
	};

	public String getName(){
		return "QL";
	};

	public String getDocumentationURL(){
		return "https://pubsonline.informs.org/doi/pdf/10.1287/opre.24.6.1164";
	};

	@Override
	public DemandAllocationResult allocateDemand(Demand demand, Network network) {
		int volume = (int) Math.ceil(demand.getVolume() / 10) - 1;

		List<PartedPath> candidatePaths = demand.getCandidatePaths(false, network);
		if (candidatePaths.isEmpty())
			return DemandAllocationResult.NO_SPECTRUM;

		candidatePaths = applyMetrics(network, volume, candidatePaths);

		if (candidatePaths.isEmpty())
			return DemandAllocationResult.NO_REGENERATORS;

		boolean workingPathSuccess = false;

		try {
			for (PartedPath path : candidatePaths)
				if (demand.allocate(network, path)) {
					workingPathSuccess = true;
					//update Q table as the demand is allocated successfully
					updateQtable(volume, path, true);
					break;
				}

		}
		catch (NetworkException storage) {
			workingPathSuccess = false;
			return DemandAllocationResult.NO_REGENERATORS;
		}
		if (!workingPathSuccess) {
			//update Q table as none of candidate paths is allocated successfully
			for (PartedPath path : candidatePaths) {
				updateQtable(volume, path, false);
			}
			return DemandAllocationResult.NO_SPECTRUM;
		}
		try {
			if (demand.allocateBackup()) {
				volume = (int) Math.ceil(demand.getSqueezedVolume() / 10) - 1;

				candidatePaths = applyMetrics(network, volume, demand.getCandidatePaths(true, network));

				if (candidatePaths.isEmpty())
					return new DemandAllocationResult(
							demand.getWorkingPath());
				for (PartedPath path : candidatePaths)
					if (demand.allocate(network, path))
						return new DemandAllocationResult(demand.getWorkingPath(), demand.getBackupPath());

				return new DemandAllocationResult(demand.getWorkingPath());
			}
		}
		catch (NetworkException e) {
			workingPathSuccess = false;
			return DemandAllocationResult.NO_REGENERATORS;
		}

		return new DemandAllocationResult(demand.getWorkingPath());
	}

	private static List<PartedPath> applyMetrics(Network network, int v, List<PartedPath> candidatePaths) {
		pathLoop: for (PartedPath path : candidatePaths) {
			path.mergeRegeneratorlessParts();

			// choosing modulations for parts
			for (PathPart part : path) {
				NetworkNode source = part.getSource();
				NetworkNode destination = part.getDestination();

				Optional<Tuple<Modulation, Double>> modulationOpt = getModulationFromQtable(source, destination, v, network.getAllowedModulations(), part);
				if (!modulationOpt.isPresent()){
					//unable to find a modulation
					path.setMetric(Integer.MAX_VALUE);
					continue pathLoop;
				}
				part.setModulation(modulationOpt.get().getX(), modulationOpt.get().getY().intValue()); //put Q value as metric


//				for (Modulation modulation : network.getAllowedModulations())
//					if (modulation.modulationDistances[volume] >= part.getLength())
//						part.setModulationIfBetter(modulation, calculateModulationMetric(network, part, modulation));

				if (part.getModulation() == null){
					path.setMetric(Integer.MAX_VALUE);
					continue pathLoop;
				}

			}
			path.calculateMetricFromParts();
			path.mergeIdenticalModulation(v);

			// Unify modulations if needed
			if (!network.canSwitchModulation()) {
				Modulation modulation = path.getModulationFromLongestPart();
				for (PathPart part : path)
					part.setModulation(modulation, calculateModulationMetric(network, part, modulation));
				path.calculateMetricFromParts();
			}

		}
		Collections.sort(candidatePaths);
		for (int i = 0; i < candidatePaths.size(); i++)
			if (candidatePaths.get(i).getMetric() == Integer.MAX_VALUE){
				candidatePaths.remove(i);
				i--;
			}

		return candidatePaths;
	}

	private static int calculateModulationMetric(Network network, PathPart part, Modulation modulation) {
		double slicesOccupationPercentage = part.getOccupiedSlicesPercentage() * 100;
		int slicesOccupationMetric;

		if (slicesOccupationPercentage <= 90)
			if (slicesOccupationPercentage <= 75)
				if (slicesOccupationPercentage <= 60)
					if (slicesOccupationPercentage <= 40)
						if (slicesOccupationPercentage <= 20)
							slicesOccupationMetric = 0;
						else
							slicesOccupationMetric = 1;
					else
						slicesOccupationMetric = 2;
				else
					slicesOccupationMetric = 3;
			else
				slicesOccupationMetric = 4;
		else
			slicesOccupationMetric = 5;

		return network.getDynamicModulationMetric(modulation, slicesOccupationMetric);
	}


	private static Optional<Tuple<Modulation,Double>> getModulationFromQtable(NetworkNode source, NetworkNode destination, int v, List<Modulation> modulations, PathPart part) {
		double random = Math.random();

		if (random < EPSILON) {
			// exploit

			INDArray qvalues = getQvalues(source, destination, v, getUsageIdx(part));
			//get the modulation with greatest Q value while fitting the volume (bit rate)
			OptionalDouble max = modulations.stream()
					.filter(m -> m.modulationDistances[v] > part.getLength())
					.mapToDouble(m -> qvalues.getDouble(getModulationId(m)))
					.max();
			if (max.isPresent()){
				return modulations.stream()
						.filter(m -> qvalues.getDouble(getModulationId(m)) == max.getAsDouble() && m.modulationDistances[v] > part.getLength())
						.max(Comparator.comparingInt(Simulation::getModulationId))
						.map(m-> {
							int l = getLinkId(
									getNodeId(source), getNodeId(destination));
							return new Tuple<>(m,
									- qTable.getDouble(l, v, getUsageIdx(part), getModulationId(m)));
						});


			} else {
				return Optional.empty();
			}

//			return modulations.stream()
//					.filter(m -> m.modulationDistances[v] > part.getLength())
//					.max(Comparator.comparingDouble(m -> qvalues.getDouble(getModulationId(m))))
//					.map(m->{
//						modulationSelected.put(m, modulationSelected.getOrDefault(m, 0L)+1);
//						return m;
//					})
//					.map(m->new Tuple<>(m,
//							- (int)Math.round(qTable.getDouble(getNodeId(source), getNodeId(destination), v, getModulationId(m)))));
		} else {
			//explore by random pick
			List<Modulation> allowedModulations = modulations.stream()
					.filter(m -> m.modulationDistances[v] > part.getLength())
					.collect(Collectors.toList());
			int pick = (int) Math.floor(Math.random() * allowedModulations.size());

			Modulation m = modulations.get(pick);
			modulationSelected.put(m, modulationSelected.getOrDefault(m, 0L)+1);
			return Optional.of(new Tuple<>(m, -qTable.maxNumber().doubleValue()));
		}

	}


	private void updateQtable(int v, PartedPath path, boolean allocateResult) {
//		if (learningCount > 10000){
//			return;
//		}
		//calculate reward
		double slicePercentageFactor = path.getParts().parallelStream()
				.mapToDouble(pathPart -> 1.0 - pathPart.getOccupiedSlicesPercentage())
				.max()
				.orElse(0);
		double reward = allocateResult ?  100 * slicePercentageFactor : NEG_REWARD;		//tested -800, -500, -1800 - seems the more -ve the reward for unallocated path, the lower the spectrum blocking %



		path.getParts().parallelStream()
				.forEach(part->{
					//update reward base on the part length as a % of max modulation distance.
					double r = reward;
					if (allocateResult) {
						double u =  ((double) part.getLength()) / part.getModulation().modulationDistances[v];
						 r = reward * u;
//					} else {
//						r = reward * (1-part.getOccupiedSlicesPercentage());
					}
					//Update Q table
					Simulation.updateQtable(r, v, part);

				});

	}


}
