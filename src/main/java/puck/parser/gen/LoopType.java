package puck.parser.gen;

import com.nativelibs4java.opencl.CLContext;

/**
 * TODO
 *
 * @author dlwh
 */
public enum LoopType {
    Inside,
    InsideNT,
    InsideTN,
    OutsideL,
    OutsideR,
    OutsideLTerm,
    OutsideRTerm;

    CLWorkQueueKernels queue(int numCoarseSymbols, CLContext context) {
        return CLWorkQueueKernels.forLoopType(numCoarseSymbols, this, context);
    }
}
