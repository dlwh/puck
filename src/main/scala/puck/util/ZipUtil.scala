package puck.util

import java.util.zip._
import java.io._
import com.nativelibs4java.opencl._
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._


object ZipUtil {
  def addEntry(out: ZipOutputStream, name: String, data: Array[Byte], compress: Boolean = true) {
    val ze = new ZipEntry(name)
    ze.setMethod(if(compress) ZipEntry.DEFLATED else ZipEntry.STORED)
    ze.setSize(data.length)
    val crc = new CRC32()
    crc.update(data,0,data.length)
    ze.setCrc(crc.getValue())
    out.putNextEntry(ze)
    out.write(data)
    out.closeEntry()
  }

  def readEntry(in: ZipFile, entry: ZipEntry):Array[Byte] = {
    val iin = in.getInputStream(entry)
    val bytes = new Array[Byte](entry.getSize.toInt)
    iin.read(bytes, 0, bytes.length)
    bytes
  }

  def serializedEntry(out: ZipOutputStream, name: String, v: AnyRef, compress: Boolean = true) {
    val barr = new ByteArrayOutputStream()
    val oout = new ObjectOutputStream(barr)
    oout.writeObject(v)
    oout.close()
    addEntry(out, name, barr.toByteArray, compress)
  }

  def deserializeEntry[T](in: InputStream) = {
    val oin = new ObjectInputStream(in)
    val t = oin.readObject().asInstanceOf[T]
    t
  }

  def addKernelSet(out: ZipOutputStream, name: String, set: IndexedSeq[CLKernel]) {
    val entries = new ArrayBuffer[String]()
    for( (k, i) <- set.zipWithIndex) {
      val ename = s"$name/$i"
      addKernel(out, ename, k)
      entries += ename
    }
    ZipUtil.serializedEntry(out, s"$name/entries", entries)
  }

  def hasKernelSet(in: ZipFile, name: String)(implicit ctxt: CLContext):Boolean = try {
     deserializeEntry[IndexedSeq[String]](in.getInputStream(in.getEntry(name+"/entries")))
     true
  } catch {
    case ex: Exception => false
  }

  def readKernelSet(in: ZipFile, name: String)(implicit ctxt: CLContext): IndexedSeq[CLKernel] = {
    val entries = deserializeEntry[IndexedSeq[String]](in.getInputStream(in.getEntry(name+"/entries")))
    for(name <- entries) yield { readKernel(in, name) }
  }


  def addKernel(out: ZipOutputStream, name: String, k: CLKernel) {
    val binaries = k.getProgram.getBinaries 
    val source = k.getProgram.getSource
    val sigs = ArrayBuffer[String]()
    for( (dev, data) <- binaries.asScala) {
      val sig = dev.createSignature
      sigs += sig
      ZipUtil.addEntry(out, s"$name.binary.$sig", data)
    }
    ZipUtil.serializedEntry(out, s"$name.sigs", sigs)
    ZipUtil.addEntry(out, s"$name.name", k.getFunctionName.getBytes("utf-8"))
    ZipUtil.addEntry(out, s"$name.source", source.getBytes("utf-8"))
  }


  def readKernel(in: ZipFile, name: String)(implicit ctxt: CLContext):CLKernel = {
    val sigs = deserializeEntry[IndexedSeq[String]](in.getInputStream(in.getEntry(name+".sigs"))).toSet
    val binaries = for(dev <- ctxt.getDevices;
      sig = dev.createSignature if  sigs(sig);
      entry <- Option(in.getEntry(s"$name.binary.$sig"))) yield {
      dev -> readEntry(in, entry)
    }

    val kname = new String(readEntry(in, in.getEntry(s"$name.name")))
    val ksource = new String(readEntry(in, in.getEntry(s"$name.source")))

    val prog = HackZipUtilHelper.newCLProgram(ctxt, binaries.toMap.asJava, ksource)
    prog.createKernel(kname)
  }
}


