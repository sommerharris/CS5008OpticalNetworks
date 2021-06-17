package ca.bcit.net;

public enum Modulation {
	
	BPSK, QPSK, QAM8, QAM16, QAM32, QAM64;
	
	//each modulation has transmission reach limit
	public final int[] modulationDistances = new int[40];
	//each modulation uses different values of slices
	public final int[] slicesConsumption = new int[40];
}
