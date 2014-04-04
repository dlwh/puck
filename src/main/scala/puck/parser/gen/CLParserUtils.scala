package puck.parser.gen

import puck.roundUpToMultipleOf

import com.nativelibs4java.opencl._
import java.util.zip.{ZipFile, ZipOutputStream}
import puck.util.ZipUtil
import puck.linalg.CLMatrix
import org.bridj.Pointer
import puck.parser.{LogSumRuleSemiring, ViterbiRuleSemiring, RuleSemiring, RuleStructure}

case class CLParserUtils(setRootScoresKernel: CLKernel,
                         getRootScoresKernel: CLKernel) {
  def write(prefix: String, out: ZipOutputStream) {
    ZipUtil.addKernel(out, s"$prefix/setRootScoresKernel", setRootScoresKernel)
    ZipUtil.addKernel(out, s"$prefix/getRootScoresKernel", getRootScoresKernel)
  }

  def setRootScores(charts: CLMatrix[Float],
                    chartIndices: CLBuffer[Integer], numUniqueParents: Int,
                    root: Int,
                    one: Float,
                    events: CLEvent*)(implicit queue: CLQueue):CLEvent = {

    setRootScoresKernel.setArgs(charts.data.safeBuffer, chartIndices,
      Integer.valueOf(numUniqueParents), Integer.valueOf(charts.rows),
      Integer.valueOf(root), java.lang.Float.valueOf(one))

    setRootScoresKernel.enqueueNDRange(queue, Array(numUniqueParents), events:_*)
  }

  def getRootScores(dest: CLBuffer[Float],
                    charts: CLMatrix[Float],
                    chartIndices: CLBuffer[Integer], numUniqueParents: Int,
                    root: Int,
                    events: CLEvent*)(implicit queue: CLQueue):CLEvent = {

    getRootScoresKernel.setArgs(dest, charts.data.safeBuffer, chartIndices,
      Integer.valueOf(numUniqueParents), Integer.valueOf(charts.rows),
      Integer.valueOf(root))

    getRootScoresKernel.enqueueNDRange(queue, Array(numUniqueParents), events:_*)
  }

}

object CLParserUtils {
  def read(prefix: String, zf: ZipFile)(implicit ctxt: CLContext) = {
    CLParserUtils(
      ZipUtil.readKernel(zf, s"$prefix/setRootScoresKernel"),
      ZipUtil.readKernel(zf, s"$prefix/getRootScoresKernel")
    )
  }

  def make[C, L](structure: RuleStructure[C, L])(implicit context: CLContext, semiring: RuleSemiring) = {

    val prog = context.createProgram(rootScoresKernel)

    CLParserUtils(
      prog.createKernel("setRootScores"),
      prog.createKernel("getRootScores"))
  }

  val rootScoresKernel = {
    """
__kernel void setRootScores(__global float* charts, __global int* indices, int numIndices, int numSyms, int root, float value) {
  int id = get_global_id(0);
  if(id < numIndices)
      charts[numSyms * indices[id] + root] = value;
}

 __kernel void getRootScores(__global float* buf, __global const float* charts, __global int* indices, int numIndices, int numSyms, int root) {
   int id = get_global_id(0);
   if(id < numIndices) {
      buf[id] = charts[numSyms * indices[id] + root];
   }
 }

                                            """



  }
}


