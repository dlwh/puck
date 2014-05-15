package puck.parser

import java.io.File
import breeze.config.CommandLineParser
import com.nativelibs4java.opencl.{JavaCL, CLContext}
import java.util.Collections
import com.typesafe.scalalogging.slf4j.LazyLogging
import epic.trees.AnnotatedLabel
import java.util.zip.ZipFile
import epic.parser.{SimpleGrammar, GenerativeParser}
import epic.parser.SimpleGrammar.CloseUnaries
import puck.parser.gen.GenType

/**
 * TODO
 *
 * @author dlwh
 **/
object CompileGrammar extends LazyLogging {
  case class Params(device: String = "nvidia",
                    grammar: File = new File("grammar.grz"),
                    textGrammarPrefix: String,
                    reproject: Boolean = false,
                    viterbi: Boolean = true,
                    logsum: Boolean = false)

  def main(args: Array[String]) = {
    val (config, files) = CommandLineParser.parseArguments(args)
    val params:Params = config.readIn[Params]("")
    import params._

    implicit val context: CLContext = {
      val (good, bad) = JavaCL.listPlatforms().flatMap(_.listAllDevices(true)).partition(d => params.device.r.findFirstIn(d.toString.toLowerCase()).nonEmpty)
      if(good.isEmpty) {
        JavaCL.createContext(Collections.emptyMap(), bad.sortBy(d => d.toString.toLowerCase.contains("geforce")).last)
      } else {
        JavaCL.createContext(Collections.emptyMap(), good.head)
      }

    }
    logger.info(s"Compiling grammar using context $context")

    val defaultGenerator = GenType.CoarseParent
    val prunedGenerator = GenType.CoarseParent

    val finePassSemiring = if(viterbi) {
      ViterbiRuleSemiring
    } else if (logsum) {
      LogSumRuleSemiring
    } else {
      RealSemiring
    }

    val parserDatas: IndexedSeq[CLParserData[AnnotatedLabel, AnnotatedLabel, String]] = {
      val paths = textGrammarPrefix.split(":")
      var grammars = paths.zipWithIndex.map{ case (f,i) => SimpleGrammar.parseBerkeleyText(f,  -12, CloseUnaries.None)}
      if(reproject && grammars.length > 1) {
        val (newc, newr) = CLParser.reprojectGrammar(grammars.head, textGrammarPrefix.split(":").head, grammars.last, textGrammarPrefix.split(":").last)
        grammars = Array(newc, newr)
      }

      val fineLayer =  CLParserData.make(grammars.last, if(grammars.length > 1) prunedGenerator else defaultGenerator, true, finePassSemiring)
      val coarseGrammars = grammars.dropRight(1)
      val coarseData = coarseGrammars.map(CLParserData.make(_,  defaultGenerator, directWrite = true, ViterbiRuleSemiring))

      coarseData :+ fineLayer
    }

    CLParserData.writeSequence(params.grammar, parserDatas)


  }

}
