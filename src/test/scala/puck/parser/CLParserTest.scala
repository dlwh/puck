package puck.parser

import com.nativelibs4java.opencl.{CLPlatform, JavaCL}
import epic.parser.SimpleRefinedGrammar
import org.scalatest.FunSuite

/**
 * TODO
 *
 * @author dlwh
 **/
class CLParserTest extends FunSuite {

  test("simple test") {
    implicit val clcontext = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU)
    val grammar = ParserTestHarness.simpleParser.augmentedGrammar.refined.asInstanceOf[SimpleRefinedGrammar[String, String, String]]
    val data = CLParserData.make(grammar)
    val parser = new CLParser(IndexedSeq(data), maxAllocSize = 40 * 1024 * 1024, profile = false)
    val parts = parser.partitions(ParserTestHarness.getTrainTrees().map(_.words))
    assert(!parts.exists(_.isInfinite) && !parts.exists(_.isNaN), parts)
  }

  test("pruning test") {
    implicit val clcontext = JavaCL.createBestContext(CLPlatform.DeviceFeature.CPU)
    val grammar = ParserTestHarness.simpleParser.augmentedGrammar.refined.asInstanceOf[SimpleRefinedGrammar[String, String, String]]
    val data = CLParserData.make(grammar)
    val parser = new CLParser(IndexedSeq(data, data), maxAllocSize = 40 * 1024 * 1024, profile = false)
    val parts = parser.partitions(ParserTestHarness.getTrainTrees().map(_.words))
    assert(!parts.exists(_.isInfinite) && !parts.exists(_.isNaN), parts)
  }

}
