package ca.bcit.net.demand;

import ca.bcit.net.PartedPath;

public class DemandAllocationResult {

	public enum Type {
		SUCCESS, NO_REGENERATORS, NO_SPECTRUM
	}
	
	public static final DemandAllocationResult NO_REGENERATORS = new DemandAllocationResult(Type.NO_REGENERATORS);
	public static final DemandAllocationResult NO_SPECTRUM = new DemandAllocationResult(Type.NO_SPECTRUM);
	
	public final Type type;
	public final PartedPath workingPath;
	private final PartedPath backupPath;
	
	private DemandAllocationResult(Type type) {
		this.type = type;
		workingPath = null;
		backupPath = null;
	}

	public DemandAllocationResult(PartedPath workingPath) {
		this.type = Type.SUCCESS;
		this.workingPath = workingPath;
		this.backupPath = null;
	}
	
	public DemandAllocationResult(PartedPath workingPath, PartedPath backupPath) {
		this.type = Type.SUCCESS;
		this.workingPath = workingPath;
		this.backupPath = backupPath;
	}
}
