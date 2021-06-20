package ca.bcit.net;

import ca.bcit.ApplicationResources;
import ca.bcit.Settings;
import ca.bcit.Tuple;
import ca.bcit.graph.Relation;
import ca.bcit.io.Logger;
import ca.bcit.io.SimulationSummary;
import ca.bcit.io.project.Project;
import ca.bcit.jfx.components.ResizableCanvas;
import ca.bcit.jfx.components.TaskReadyProgressBar;
import ca.bcit.jfx.controllers.MainWindowController;
import ca.bcit.jfx.controllers.SimulationMenuController;
import ca.bcit.jfx.tasks.SimulationTask;
import ca.bcit.net.algo.QL;
import ca.bcit.net.demand.AnycastDemand;
import ca.bcit.net.demand.Demand;
import ca.bcit.net.demand.DemandAllocationResult;
import ca.bcit.net.demand.generator.TrafficGenerator;
import ca.bcit.net.spectrum.Spectrum;
import ca.bcit.utils.LocaleUtils;
import ca.bcit.utils.collections.HashArray;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import javafx.fxml.FXMLLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;


/**
 * Main simulation class (start point)
 */
public class Simulation {
    public static final int LEARNING_COUNT = 10000;
    public static INDArray qTable = null;
    public static final int NUM_OF_VOLUMES = 40;
    public static final double DISCOUNT_FACTOR = 0.8;
    public static final double LEARNING_RATE = 0.9;
    public static final double EPSILON = 0.5;
    public static Map<NetworkNode, Integer> networkNodes = null;
    public static int learningCount = 0;
    public static Map<Modulation, Long> modulationSelected;


    public static final String RESULTS_DATA_DIR_NAME = "results data";
    private String resultsDataFileName;

    private Network network;
    private TrafficGenerator generator;
    private double totalVolume;
    private double spectrumBlockedVolume;
    private double regeneratorsBlockedVolume;
    private double linkFailureBlockedVolume;
    private double regsPerAllocation;
    private double allocations;
    private double unhandledVolume;
    private final double[] modulationsUsage = new double[6];
    private boolean multipleSimulations = false;
    public static Map<Tuple<Integer, Integer>, Integer> linkMap;

    public Simulation() {
    }

    public Simulation(Network network, TrafficGenerator generator) {
        this.network = network;
        this.generator = generator;
    }

    public Simulation(Network network, TrafficGenerator generator, boolean printSummary) {
        this.network = network;
        this.generator = generator;
    }

    public Simulation(Network network, TrafficGenerator generator, boolean printSummary, int totalSimulations, int startingErlangValue, int currentErlangValue, int endingErlangValue, int randomSeed, double alpha) {
        this.network = network;
        this.generator = generator;
    }

    public void simulate(long seed, int demandsCount, double alpha, int erlang, boolean replicaPreservation, SimulationTask task) {
        SimulationMenuController.finished = false;
        SimulationMenuController.cancelled = false;
        clearVolumeValues();

        /* Q Learning - start */
        //initialize Q values
        initQtable(network);
        learningCount = 0;
        modulationSelected = new HashMap<>();
        /* Q Learning - end */
        int i = 0;

        //For development set to debug, for release set to info
        Logger.setLoggerLevel(Logger.LoggerLevel.DEBUG);
        generator.setErlang(erlang);
        generator.setSeed(seed);
        generator.setReplicaPreservation(replicaPreservation);
        network.setSeed(seed);
        Random linkCutter = new Random(seed);

        try {
            ResizableCanvas.getParentController().updateGraph();

            for (; generator.getGeneratedDemandsCount() < demandsCount; ) {
                SimulationMenuController.started = true;

                Demand demand = generator.next();

                // handle the demand for the specific simulation
                if (linkCutter.nextDouble() < alpha / erlang)
                    for (Demand reallocate : network.cutLink())
                        if (reallocate.reallocate())
                            handleDemand(reallocate);
                        else {
                            linkFailureBlockedVolume += reallocate.getVolume();
                            ResizableCanvas.getParentController().linkFailureBlockedVolume += reallocate.getVolume();
                        }
                else {
                    handleDemand(demand);
                    if (demand instanceof AnycastDemand)
                        handleDemand(generator.next());
                }

                network.update();

                // pause button
                pause();

                // cancel button
                if (SimulationMenuController.cancelled) {
                    Logger.info(LocaleUtils.translate("simulation_cancelled"));
                    break;
                }

                task.updateProgress(generator.getGeneratedDemandsCount(), demandsCount);
                learningCount++;

            } // loop end here
            try (FileWriter w = new FileWriter(String.format("qtable-end-%d-er%d-req%d-reward%d.txt",
                    new Date().getTime(), erlang, demandsCount, QL.NEG_REWARD))) {
                w.write(qTable.toStringFull());
            } catch (IOException e) {
                e.printStackTrace();
            }
            Logger.info("Test: " + modulationSelected.toString());
            // force call the update again here
        } catch (NetworkException e) {


            //TODO should exclude the first 100,000 (say) due to q learning
            Logger.info(LocaleUtils.translate("network_exception_label") + " " + LocaleUtils.translate(e.getMessage()));
            for (; generator.getGeneratedDemandsCount() < demandsCount; ) {
                Demand demand = generator.next();
                if (i > LEARNING_COUNT) {
                    unhandledVolume += demand.getVolume();
                }
                if (demand instanceof AnycastDemand)
                    if (i > LEARNING_COUNT) {
                        unhandledVolume += generator.next().getVolume();
                    }
                task.updateProgress(generator.getGeneratedDemandsCount(), demandsCount);
            }
            totalVolume += unhandledVolume;
            ResizableCanvas.getParentController().totalVolume += unhandledVolume;
            i++;
            System.out.println("i=" + i);

        }


        //wait for internal cleanup after simulation is done
        network.waitForDemandsDeath();
        ResizableCanvas.getParentController().stopUpdateGraph();
        ResizableCanvas.getParentController().resetGraph();

        // signal GUI menus that simulation is complete
        SimulationMenuController.finished = true;
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/ca/bcit/jfx/res/views/SimulationMenu.fxml"), Settings.getCurrentResources());
        SimulationMenuController simulationMenuController = fxmlLoader.<SimulationMenuController>getController();
        if (simulationMenuController != null)
            simulationMenuController.disableClearSimulationButton();

        Logger.info(LocaleUtils.translate("blocked_spectrum_label") + " " + (spectrumBlockedVolume / totalVolume) * 100 + "%");
        Logger.info(LocaleUtils.translate("blocked_regenerators_label") + " " + (regeneratorsBlockedVolume / totalVolume) * 100 + "%");
        Logger.info(LocaleUtils.translate("blocked_link_failure_label") + " " + (linkFailureBlockedVolume / totalVolume) * 100 + "%");

        // write the resulting data of a successful simulation to file
        File resultsDirectory = new File(RESULTS_DATA_DIR_NAME);
        if (!resultsDirectory.isDirectory())
            resultsDirectory.mkdir();
        File resultsProjectDirectory = new File(RESULTS_DATA_DIR_NAME + "/" + ApplicationResources.getProject().getName().toUpperCase());
        if (isMultipleSimulations())
            if (!resultsProjectDirectory.isDirectory())
                resultsProjectDirectory.mkdir();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(new SimulationSummary(generator.getName(), erlang, seed, alpha, demandsCount, totalVolume,
                spectrumBlockedVolume, regeneratorsBlockedVolume, linkFailureBlockedVolume, unhandledVolume, regsPerAllocation,
                allocations, network.getDemandAllocationAlgorithm().getName()));
        try {
            resultsDataFileName = ApplicationResources.getProject().getName().toUpperCase() +
                    new SimpleDateFormat("_yyyy_MM_dd_HH_mm_ss").format(new Date()) + ".json";
            TaskReadyProgressBar.addResultsDataFileName(resultsDataFileName);
            FileWriter resultsDataWriter = new FileWriter(new File(
                    isMultipleSimulations() ? resultsProjectDirectory : resultsDirectory, resultsDataFileName));

            resultsDataWriter.write(json);
            resultsDataWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        //Helps slow GUI update between multiple simulations being run back to back
        try {
            Thread.sleep(2000);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
    }

    public void simulate(long seed, int demandsCount, double alpha, int erlang, boolean replicaPreservation) {
        SimulationMenuController.finished = false;
        SimulationMenuController.cancelled = false;
        clearVolumeValues();

        /* Q Learning - start */
        //initialize Q values
        initQtable(network);
        learningCount = 0;
        modulationSelected = new HashMap<>();
        /* Q Learning - end */
        int i = 0;


        ResourceBundle resourceBundle = ResourceBundle.getBundle("ca.bcit.bundles.lang", LocaleUtils.getLocaleFromLocaleEnum(Settings.CURRENT_LOCALE));

        //For development set to debug, for release set to info
        Logger.setLoggerLevel(Logger.LoggerLevel.DEBUG);
        generator.setErlang(erlang);
        generator.setSeed(seed);
        generator.setReplicaPreservation(replicaPreservation);
        network.setSeed(seed);
        Random linkCutter = new Random(seed);

        try {
            ResizableCanvas.getParentController().updateGraph();

            for (; generator.getGeneratedDemandsCount() < demandsCount; ) {
                SimulationMenuController.started = true;

                Demand demand = generator.next();

                // handle the demand for the specific simulation
                if (linkCutter.nextDouble() < alpha / erlang)
                    for (Demand reallocate : network.cutLink())
                        if (reallocate.reallocate())
                            handleDemand(reallocate);
                        else {
                            linkFailureBlockedVolume += reallocate.getVolume();
                            ResizableCanvas.getParentController().linkFailureBlockedVolume += reallocate.getVolume();
                        }
                else {
                    handleDemand(demand);
                    if (demand instanceof AnycastDemand)
                        handleDemand(generator.next());
                }

                network.update();

                // pause button
                pause();

                // cancel button
                if (SimulationMenuController.cancelled) {
                    Logger.info(LocaleUtils.translate("simulation_cancelled"));
                    break;
                }
                learningCount++;
            } // loop end here
            try (FileWriter w = new FileWriter(String.format("qtable-end-%d-er%d-req%d-reward%d.txt",
                    new Date().getTime(), erlang, demandsCount, QL.NEG_REWARD))) {
                w.write(qTable.toStringFull());
            } catch (IOException e) {
                e.printStackTrace();
            }
            Logger.info("Test: " + modulationSelected.toString());
            // force call the update again here
        } catch (NetworkException e) {


            //TODO should exclude the first 100,000 (say) due to q learning
            Logger.info(LocaleUtils.translate("network_exception_label") + " " + LocaleUtils.translate(e.getMessage()));
            for (; generator.getGeneratedDemandsCount() < demandsCount; ) {
                Demand demand = generator.next();
                if (i > LEARNING_COUNT) {
                    unhandledVolume += demand.getVolume();
                }
                if (demand instanceof AnycastDemand)
                    if (i > LEARNING_COUNT) {
                        unhandledVolume += generator.next().getVolume();
                    }
            }
            totalVolume += unhandledVolume;
            ResizableCanvas.getParentController().totalVolume += unhandledVolume;
            i++;
            Logger.info("i=" + i);
//            System.out.println("i=" + i);
        }
        String algorithm = network.getDemandAllocationAlgorithm().getName();

        //wait for internal cleanup after simulation is done
        network.waitForDemandsDeath();
        ResizableCanvas.getParentController().stopUpdateGraph();
        ResizableCanvas.getParentController().resetGraph();

        // signal GUI menus that simulation is complete
        SimulationMenuController.finished = true;
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/ca/bcit/jfx/res/views/SimulationMenu.fxml"), resourceBundle);
        SimulationMenuController simulationMenuController = fxmlLoader.<SimulationMenuController>getController();
        if (simulationMenuController != null)
            simulationMenuController.disableClearSimulationButton();

        Logger.info(LocaleUtils.translate("blocked_spectrum_label") + " " + (spectrumBlockedVolume / totalVolume) * 100 + "%");
        Logger.info(LocaleUtils.translate("blocked_regenerators_label") + " " + (regeneratorsBlockedVolume / totalVolume) * 100 + "%");
        Logger.info(LocaleUtils.translate("blocked_link_failure_label") + " " + (linkFailureBlockedVolume / totalVolume) * 100 + "%");

        // write the resulting data of a successful simulation to file
        File resultsDirectory = new File(RESULTS_DATA_DIR_NAME);
        if (!resultsDirectory.isDirectory())
            resultsDirectory.mkdir();
        File resultsProjectDirectory = new File(RESULTS_DATA_DIR_NAME + "/" + ApplicationResources.getProject().getName().toUpperCase());
        if (isMultipleSimulations())
            if (!resultsProjectDirectory.isDirectory())
                resultsProjectDirectory.mkdir();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(new SimulationSummary(generator.getName(), erlang, seed, alpha, demandsCount, totalVolume,
                spectrumBlockedVolume, regeneratorsBlockedVolume, linkFailureBlockedVolume, unhandledVolume, regsPerAllocation,
                allocations, algorithm));
        try {
            resultsDataFileName = ApplicationResources.getProject().getName().toUpperCase() +
                    new SimpleDateFormat("_yyyy_MM_dd_HH_mm_ss").format(new Date()) + ".json";
            TaskReadyProgressBar.addResultsDataFileName(resultsDataFileName);
            FileWriter resultsDataWriter = new FileWriter(new File(isMultipleSimulations() ? resultsProjectDirectory : resultsDirectory, resultsDataFileName));

            resultsDataWriter.write(json);
            resultsDataWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        //Helps slow GUI update between multiple simulations being run back to back
        try {
            Thread.sleep(2000);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * For use during an active simulation only. Place the simulation thread to sleep while pause is active.
     */
    private void pause() {
        while (SimulationMenuController.paused) {
            try {
                Thread.sleep(10);
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Reset parameters to be used in a new simulation. Called before a set of simulations start.
     */
    private void clearVolumeValues() {
        MainWindowController mainWindowController = ResizableCanvas.getParentController();
        Project project = ApplicationResources.getProject();
        Network network = project.getNetwork();
        this.totalVolume = 0;
        this.spectrumBlockedVolume = 0;
        this.regeneratorsBlockedVolume = 0;
        this.linkFailureBlockedVolume = 0;
        this.regsPerAllocation = 0;
        this.allocations = 0;
        this.unhandledVolume = 0;
        mainWindowController.totalVolume = 0;
        mainWindowController.spectrumBlockedVolume = 0;
        mainWindowController.regeneratorsBlockedVolume = 0;
        mainWindowController.linkFailureBlockedVolume = 0;
        for (NetworkNode n : network.getNodes()) {
            n.clearOccupied();
            for (NetworkNode n2 : network.getNodes()) {
                if (network.containsLink(n, n2)) {
                    NetworkLink networkLink = network.getLink(n, n2);
                    Spectrum spectrum = network.getLinkSlices(n, n2);
                    networkLink.slicesUp = new Spectrum(NetworkLink.NUMBER_OF_SLICES);
                    networkLink.slicesDown = new Spectrum(NetworkLink.NUMBER_OF_SLICES);
                }
            }
        }
    }

    /**
     * Process a specific demand request. If the demand is impossible to fulfill, the cause is recorded.
     * If the demand can be fulfilled, resources will be consumed.
     *
     * @param demand the demand in question
     */
    private void handleDemand(Demand demand) {
        DemandAllocationResult result = network.allocateDemand(demand);

        if (result.workingPath == null)
            switch (result.type) {
                case NO_REGENERATORS:
                    regeneratorsBlockedVolume += demand.getVolume();
                    ResizableCanvas.getParentController().regeneratorsBlockedVolume += demand.getVolume();
                    break;
                case NO_SPECTRUM:
                    spectrumBlockedVolume += demand.getVolume();
                    ResizableCanvas.getParentController().spectrumBlockedVolume += demand.getVolume();
                    break;
                default:
                    break;
            }
        else {
            allocations++;
            regsPerAllocation += demand.getWorkingPath().getPartsCount() - 1;
            if (demand.getBackupPath() != null)
                regsPerAllocation += demand.getBackupPath().getPartsCount() - 1;
            double modulationsUsage[] = new double[6];
            for (PathPart part : result.workingPath)
                modulationsUsage[part.getModulation().ordinal()]++;
            for (int i = 0; i < 6; i++) {
                modulationsUsage[i] /= result.workingPath.getPartsCount();
                this.modulationsUsage[i] += modulationsUsage[i];
            }
        }
        totalVolume += demand.getVolume();
        ResizableCanvas.getParentController().totalVolume += demand.getVolume();
    }

    public boolean isMultipleSimulations() {
        return multipleSimulations;
    }

    public void setMultipleSimulations(boolean multipleSimulations) {
        this.multipleSimulations = multipleSimulations;
    }

    // Q Learning - start
    private static void initQtable(Network network) {
        initNetworkNodeMap(network);

        List<Modulation> allowedModulations = network.getAllowedModulations();
//        int networkNodeSize = networkNodes.size();

        //initialize Q table to have values -100000.
        //qTable[from][to][v][link-usage%][m]
        //e.g. from node 3 to node 10, v= 0,1,2,... 40 for 10Gb/s, 20, 30... 400 , m = QAM16,link usage = 50%
        //qTable[3][10][0][5][3]
//        qTable = Nd4j.zeros(networkNodeSize, networkNodeSize, NUM_OF_VOLUMES, 10, allowedModulations.size())
//                .add(-100000.0);


        HashArray<Relation<NetworkNode, NetworkLink, NetworkPath>> allLinks = network.getAllLinks();


        //assign initial Q values for each possible link
        //for each link
//        for (Iterator<Relation<NetworkNode, NetworkLink, NetworkPath>> iterator = allLinks.iterator(); iterator.hasNext(); ) {
//            Relation<NetworkNode, NetworkLink, NetworkPath> relation = iterator.next();
//            NetworkNode nodeA = relation.nodeA;
//            NetworkNode nodeB = relation.nodeB;
//            int a = getNodeId(nodeA);
//            if (a < 0) {
//                continue;
//            }
//            int b = getNodeId(nodeB);
//            if (b < 0) {
//                continue;
//            }
//
//            //for each available volume, v = volumn / 10 - 1, e.g volume = 10 => v = 0, volume = 400 => v = 39
//            for (int v = 0; v < 40; v++) {
//                for (int u = 0; u < 10; u++) {
//                    //for each modulation method
//                    for (Modulation m : allowedModulations) {
//                        //qTable[a][b][v][m] = 0, from node A to node B
//                        qTable.putScalar(new int[]{a, b, v, u, getModulationId(m)}, 0);
//                        //qTable[b][a][v][m] = 0, from node B to node A
//                        qTable.putScalar(new int[]{b, a, v, u, getModulationId(m)}, 0);
//                    }
//                }
//            }
//
//
//        }

        linkMap = new HashMap<>();
        int l = 0;
        for (Relation<NetworkNode, NetworkLink, NetworkPath> r : allLinks) {
            NetworkNode nodeA = r.nodeA;
            NetworkNode nodeB = r.nodeB;
            int a = getNodeId(nodeA);
            int b = getNodeId(nodeB);
            linkMap.put(new Tuple<>(a, b), l++);
//            linkMap.put(new Tuple<>(b, a), l++);
        }

        qTable = Nd4j.zeros(l, NUM_OF_VOLUMES, 10, allowedModulations.size());


        try (FileWriter w = new FileWriter("qtable-start.txt")) {
            w.write(qTable.toStringFull());
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static void initNetworkNodeMap(Network network) {
        NetworkNode[] nodes = network.nodes.values().toArray(new NetworkNode[0]);
        networkNodes = new HashMap<>();
        for (int i = 0; i < nodes.length; i++) {
            networkNodes.put(nodes[i], i);
        }
    }

    public static void updateQtable(double reward, int v, PathPart part) {
        NetworkNode source = part.source;
        NetworkNode destination = part.getDestination();
        Modulation modulation = part.getModulation();

        //calculate Q value
        int s = getNodeId(source);
        int d = getNodeId(destination);
        int usageIdx = getUsageIdx(part);
        double oldQ = qTable.getDouble(getLinkId(s, d), v, usageIdx, getModulationId(modulation)); //qTable[s][d][v][m]
        double temporalDifference = reward + DISCOUNT_FACTOR * qTable.maxNumber().doubleValue() - oldQ;
        double newQ = oldQ + LEARNING_RATE * temporalDifference;

        //qtable[source][destination][volume][modulation] = new value

        int[] index = {getLinkId(s, d), v, usageIdx, getModulationId(modulation)};
        qTable.putScalar(index, newQ);
    }

    public static int getLinkId(int s, int d) {
        Tuple<Integer, Integer> key = new Tuple<>(s, d);
        if (linkMap.containsKey(key)){
            return linkMap.get(key);
        }
        Tuple<Integer, Integer> key1 = new Tuple<>(d, s);
        if (linkMap.containsKey(key1)){
            return linkMap.get(key1);
        }
        throw new IllegalArgumentException();
    }

    public static int getUsageIdx(PathPart part) {
        int x = (int) Math.floor(part.getOccupiedSlicesPercentage() * 10);
        return Math.min(x, 9);
    }

    public static int getModulationId(Modulation m) {
        return Arrays.binarySearch(Modulation.values(), m);
    }

    public static int getNodeId(NetworkNode node) {
        return networkNodes.getOrDefault(node, -1);
    }

    public static INDArray getQvalues(NetworkNode source, NetworkNode destination, int v, long u) {
        int s = getNodeId(source);
        int d = getNodeId(destination);
        Tuple<Integer, Integer> key = new Tuple<>(s, d);
        int link = getLinkId(s,d);

        if (link<0 || link >= linkMap.size()){
            System.out.printf("Invalid link: %d, %d", s, d);
        }
        return qTable.get(NDArrayIndex.point(link),
                NDArrayIndex.point(v),
                NDArrayIndex.point(u),
                NDArrayIndex.all());
    }
    // Q Learning - end
}
