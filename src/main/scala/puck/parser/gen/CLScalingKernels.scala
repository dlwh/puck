package puck.parser.gen

import puck.linalg.CLMatrix
import com.nativelibs4java.opencl._
import org.bridj.Pointer
import java.util.zip.{ZipFile, ZipOutputStream}
import puck.util.ZipUtil
import scala.Array
import puck.PointerFreer
import puck.parser.{SymId, RuleSemiring, RuleStructure}

/**
 * TODO
 *
 * @author dlwh
 **/
case class CLScalingKernels(getScalingKernel: CLKernel) {


  def write(prefix: String, out: ZipOutputStream) {
    ZipUtil.addKernel(out, s"$prefix/getScalingConstantsKernel", getScalingKernel)
  }

  def getScaling(scaleConstants: CLBuffer[Float],
                 charts: CLMatrix[Float],
                 events: CLEvent*)(implicit queue: CLQueue):CLEvent = {


    getScalingKernel.setArgs(scaleConstants, charts.data.safeBuffer, Integer.valueOf(charts.rows), Integer.valueOf(charts.cols))
    val ev = getScalingKernel.enqueueNDRange(queue, Array(charts.cols, 1), Array(1, 1), events:_*)
    ev
  }

}


object CLScalingKernels {
  def read(prefix: String, zf: ZipFile)(implicit ctxt: CLContext) = {
    CLScalingKernels(ZipUtil.readKernel(zf, s"$prefix/getScalingConstantsKernel"))
  }

  def make[C, L](structure: RuleStructure[C, L])(implicit context: CLContext, semiring: RuleSemiring) = {
    val prog = context.createProgram(programText)

    CLScalingKernels(prog.createKernel("getScalingConstants"))
  }



  def programText[L, C]: String = {
    """
__kernel void getScalingConstants(__global float* scaling,
                           __global const float* chart, int numSyms,
                           int numCells) {
  const int cell = get_global_id(0);

  if(cell >= numCells) return;


   float m = -100000.0f;
    for(int sym = 0; sym < numSyms; ++sym) {
      float score = chart[numSyms * cell + sym];
      m = max(score, m);

    }
    if(m < -9000.0f)
      m = 0;

    scaling[cell] = m;
}
    """
  }
}
