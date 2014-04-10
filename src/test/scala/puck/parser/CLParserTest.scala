package puck.parser

import com.nativelibs4java.opencl.{CLPlatform, JavaCL}
import epic.parser.SimpleRefinedGrammar
import org.scalatest.FunSuite
import puck.parser.gen.GenType

/**
 * TODO
 *
 * @author dlwh
 **/
class CLParserTest extends FunSuite {

  test("simple test") {
    implicit val clcontext = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU)
    val grammar = ParserTestHarness.grammar
    val data = CLParserData.make(grammar, GenType.CoarseParent, true, ViterbiRuleSemiring)
    val parser = new CLParser(IndexedSeq(data), maxAllocSize = 40 * 1024 * 1024, profile = false)
    val parts = parser.partitions(ParserTestHarness.getTrainTrees().take(100).map(_.words))
    assert(!parts.exists(_.isInfinite) && !parts.exists(_.isNaN), parts)
  }

  test("pruning test") {
    implicit val clcontext = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU)
    println("pruning " + clcontext)
    val grammar = ParserTestHarness.grammar
    val data = CLParserData.make(grammar, GenType.CoarseParent, true, ViterbiRuleSemiring)
    val parser = new CLParser(IndexedSeq(data, data), maxAllocSize = 40 * 1024 * 1024, profile = false)
    val parts = parser.partitions(ParserTestHarness.getTrainTrees().take(100).map(_.words))
    assert(!parts.exists(_.isInfinite) && !parts.exists(_.isNaN), parts)
  }

}
