package ca.bcit.net.algo;

import ca.bcit.Tuple;
import ca.bcit.net.*;
import ca.bcit.net.demand.Demand;
import ca.bcit.net.demand.DemandAllocationResult;
import ca.bcit.net.spectrum.Spectrum;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.stream.Collectors;

import static ca.bcit.net.Simulation.*;

public class QL implements IRMSAAlgorithm{

	public static int negativeReward = -3500;

	public String getKey(){
		return "QL";
	}

	public String getName(){
		return "QL";
	}

	public String getDocumentationURL(){
		return "https://pubsonline.informs.org/doi/pdf/10.1287/opre.24.6.1164";
	}

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
					successCount ++;
					//update Q table as the demand is allocated successfully
					updateQtableWhenSuccessful(volume, path, network.getAllowedModulations());
					break;
				}

		}
		catch (NetworkException storage) {
			workingPathSuccess = false;
			return DemandAllocationResult.NO_REGENERATORS;
		}
		if (!workingPathSuccess) {
			//update Q table as none of candidate paths is allocated successfully
			List<Modulation> allowedModulations = network.getAllowedModulations();
//			if (learningCount< 10000) { //TODO consider remove
				for (PartedPath path : candidatePaths) {
					updateQtableWhenFailed(volume, path, allowedModulations);
				}
//			}
			failCount++;
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
//		if (learningCount==5000){
//			epsilon = 0.9;
//		}
		if (random < epsilon) {
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
			return Optional.of(new Tuple<>(m,
//					-qTable.getDouble(getLinkId(
//							getNodeId(source), getNodeId(destination)), v, getUsageIdx(part), getModulationId(m))
					-qTable.maxNumber().doubleValue()
			));
		}

	}
	public static LinkedList<Double> positiveRewards = new LinkedList<>();
	private void updateQtableWhenSuccessful(int v, PartedPath path, List<Modulation> modulations){
		if (learningCount>8000){
			return;
		}
		double reward = 100 * slicePercentageFactor(path);
		path.getParts().stream()
				.forEach(part->{
					rewardSuccessful(v, modulations, reward, part);
				});
	}

	private void rewardSuccessful(int v, List<Modulation> modulations, double reward, PathPart part) {
		Modulation modulation = part.getModulation();

		countModulation(modulationCount, modulation);

		double r = reward * ((double) part.getLength()) / part.getModulation().modulationDistances[v];
		int sliceConsumption = part.getModulation().slicesConsumption[v];
		Spectrum slices = part.getSlices();
		Optional<Modulation> minSlicesModulation = modulations.stream()
				.filter(m -> part.getModulation() != m
						&& m.slicesConsumption[v] < sliceConsumption
						&& slices.canAllocateWorking(m.slicesConsumption[v]) != -1)
				.min(Comparator.comparing(m -> m.slicesConsumption[v]));
		if (minSlicesModulation.isPresent()){
//						r *= ((double) minSlicesModulation.get().slicesConsumption[v]) / sliceConsumption;
			r = negativeReward * 2;
		}
		//Update Q table
		Simulation.updateQtable(r, v, part);
		positiveRewards.add(r);
	}

	private double slicePercentageFactor(PartedPath path) {

		return path.getParts().stream()
				.mapToDouble(pathPart -> 1.0 - pathPart.getOccupiedSlicesPercentage())
				.max()
				.orElse(0);
	}

	public static LinkedList<Number> negativeRewards = new LinkedList<>();
	private void updateQtableWhenFailed(int v, PartedPath path, List<Modulation> modulations){
		if (learningCount>8000){
			return;
		}
		double successfulReward = 100 * slicePercentageFactor(path);
		for(PathPart part: path.getParts()){
			Spectrum slices = part.getSlices();
			int slicesCount, offset;
			slicesCount = part.getModulation().slicesConsumption[v];
			offset = slices.canAllocateWorking(slicesCount);

			if (offset == -1){//cannot allocate
				countModulation(modulationFailed, part.getModulation());
				long possibleAllocationCount
						= modulations.stream()
						.filter(m -> part.getModulation()!=m
							 &&	slices.canAllocateWorking(m.slicesConsumption[v]) != -1
						)
						.count();
//				double discount = 0.5 + 0.5 * possibleAllocationCount / modulations.size() ;

				double reward = possibleAllocationCount > 0 ?  negativeReward * 2 : negativeReward;
				Simulation.updateQtable(reward, v, part);
				negativeRewards.add(reward);

				break;
			} else {

				rewardSuccessful(v, modulations, successfulReward, part);
//				Simulation.updateQtable(negativeReward, v, part);
				positiveRewards.add(successfulReward);
			}


		}

	}



}
