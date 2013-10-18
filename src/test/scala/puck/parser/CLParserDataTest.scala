package puck.parser

import org.scalatest.FunSuite
import com.nativelibs4java.opencl.{CLPlatform, JavaCL}
import epic.trees._
import epic.parser._
import epic.trees.TreeInstance
import epic.trees.annotations.Xbarize
import scala.collection.mutable.ArrayBuffer
import java.io.{FileOutputStream, File, ByteArrayInputStream, ByteArrayOutputStream}
import java.util.zip.ZipFile

/**
 * TODO
 *
 * @author dlwh
 **/
class CLParserDataTest extends FunSuite {
  test("write/read works") {
    implicit val clcontext = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU)
    val grammar = ParserTestHarness.simpleParser.augmentedGrammar.refined.asInstanceOf[SimpleRefinedGrammar[String, String, String]]
    val data = CLParserData.make(grammar)
    val tempFile = File.createTempFile("xxx","parserdata")
    tempFile.deleteOnExit()
    val out = new FileOutputStream(tempFile)
    data.write(out)
    out.close()
    val in = new ZipFile(tempFile)
    val input = CLParserData.read(in)
    assert(data.grammar.signature === input.grammar.signature)
  }

  /*
  test("debuggin") {
    implicit val clcontext = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU)
    val ap = clcontext.createProgram("__kernel void add(int a, int b, __global int* c) { *c = a + b; }")
    ap.createKernel("add")
    val addBins = ap.getBinaries
    val subBins = clcontext.createProgram("__kernel void sub(int a, int b, __global int* c) { *c = a - b; }").getBinaries
    val ap2 = clcontext.createProgram(addBins, "__kernel void add(int a, int b, __global int* c) { *c = a + b; }")
    ap2.createKernel("add")
    val sp2 = clcontext.createProgram(subBins, "__kernel void sub(int a, int b, __global int* c) { *c = a - b; }")
   sp2.setCached(true)
    sp2.createKernel("sub")
  }
  */
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
    massageTrees(TstTreebank.treebank.train.trees,  maxLength).map(ti => ti.copy(tree=UnaryChainRemover.removeUnaryChains(ti.tree)))
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
    ParseEval.evaluate(testTrees, parser, AnnotatedLabelChainReplacer, asString = {(_:AnnotatedLabel).baseLabel}, nthreads= -1)
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
  val simpleParser: SimpleChartParser[AnnotatedLabel, String] = {
    val trees = getTrainTrees()
    val grammar = GenerativeParser.extractGrammar[AnnotatedLabel, String](trees.head.label.label, trees.map(_.mapLabels(_.baseAnnotatedLabel)))
    SimpleChartParser(AugmentedGrammar.fromRefined(grammar))
  }
}