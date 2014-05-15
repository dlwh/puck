package puck.parser

import org.scalatest.FunSuite
import com.nativelibs4java.opencl.{CLProgram, CLPlatform, JavaCL}
import epic.trees._
import epic.parser._
import epic.trees.TreeInstance
import epic.trees.annotations.Xbarize
import scala.collection.mutable.ArrayBuffer
import java.io._
import java.util.zip.{ZipOutputStream, ZipFile}
import epic.trees.StandardTreeProcessor
import epic.trees.TreeInstance
import epic.trees.annotations.Xbarize
import java.util
import puck.parser.gen.GenType

/**
 * TODO
 *
 * @author dlwh
 **/
class CLParserDataTest extends FunSuite {
  test("write/read works") {
    implicit val clcontext = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU)
    val grammar = ParserTestHarness.grammar.asInstanceOf[SimpleGrammar[String, String, String]]
    val data = CLParserData.make(grammar, GenType.CoarseParent, false, ViterbiRuleSemiring)
    val tempFile = File.createTempFile("xxx","parserdata")
    tempFile.deleteOnExit()
    val out = new ZipOutputStream(new FileOutputStream(tempFile))
    data.write("", out)
    out.close()
    val in = new ZipFile(tempFile)
    val input = CLParserData.read("", in)
    assert(data.grammar.signature === input.grammar.signature)
  }

  test("debuggin") {
    implicit val clcontext = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU)
    implicit val cpucontext = JavaCL.createBestContext(CLPlatform.DeviceFeature.CPU)

    val src = "__kernel void add(int a, int b, __global int* c) { *c = a + b; }"
    val ap = clcontext.createProgram(src)
    val tempFile = File.createTempFile("xxx","parserdata")
    tempFile.deleteOnExit()
    val out = new FileOutputStream(tempFile)
    CLProgram.writeBinaries(ap.getBinaries, src, "XXX", out)
    out.close()
    val in = new FileInputStream(tempFile)
    // NPE in CLProgram here:
    // devices is null if the signature isn't present.
    //      List<CLDevice> devices = devicesBySignature.get(signature);
    // for (CLDevice device : devices)
    // ret.put(device, data);
    CLProgram.readBinaries(util.Arrays.asList(cpucontext.getDevices:_*), "XXX", in)
    in.close()
  }
}

object TstTreebank {
  val treebank =  {
    val train = TstTreebank.getClass.getClassLoader.getResource("smallbank/train")
    val test = TstTreebank.getClass.getClassLoader.getResource("smallbank/test")
    val dev = TstTreebank.getClass.getClassLoader.getResource("smallbank/dev")

    new SimpleTreebank(Map("train"->train),Map("dev"->dev),Map("test"->test))
  }
}


object ParserTestHarness {

  def getTrainTrees(maxLength:Int= 15): IndexedSeq[TreeInstance[AnnotatedLabel, String]] = {
    massageTrees(TstTreebank.treebank.train.trees,  maxLength).map(ti => ti.copy(tree=UnaryChainCollapser.collapseUnaryChains(ti.tree)))
  }

  def getTestTrees(maxLength:Int= 15): IndexedSeq[TreeInstance[AnnotatedLabel, String]] = {
    massageTrees(TstTreebank.treebank.test.trees, maxLength)
  }

  def massageTrees(trees: Iterator[(Tree[String], IndexedSeq[String])], maxLength:Int=15): IndexedSeq[TreeInstance[AnnotatedLabel, String]] = {
    val trainTrees = ArrayBuffer() ++= (for( (tree, words) <- trees.filter(_._2.length <= maxLength))
    yield TreeInstance("", Xbarize() apply (transform(tree), words), words))

    trainTrees
  }


  def evalParser(testTrees: IndexedSeq[TreeInstance[AnnotatedLabel, String]], parser: Parser[AnnotatedLabel, String]) = {
    ParseEval.evaluate(testTrees, parser, asString = {(_:AnnotatedLabel).baseLabel}, nthreads= -1)
  }

  val transform = new StandardTreeProcessor()

  val (simpleLexicon, simpleGrammar) = {
    try {
      val trees = getTrainTrees()
      GenerativeParser.extractLexiconAndGrammar(trees.map(_.mapLabels(_.baseAnnotatedLabel)))
    } catch {
      case e:Exception => e.printStackTrace(); throw e
    }
  }
  val grammar: SimpleGrammar[AnnotatedLabel, AnnotatedLabel, String] = {

    val trees = getTrainTrees()
    GenerativeParser.extractGrammar[AnnotatedLabel, String](trees.head.label.label, trees.map(_.mapLabels(_.baseAnnotatedLabel)))
  }
  val simpleParser: Parser[AnnotatedLabel, String] = {
    Parser(grammar)
  }
}