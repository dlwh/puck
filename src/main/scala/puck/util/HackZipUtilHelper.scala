package com.nativelibs4java.opencl

object HackZipUtilHelper {
  def newCLProgram(context: CLContext, binaries: java.util.Map[CLDevice, Array[Byte]], source: String) = {
    new CLProgram(context, binaries, source)
  }
}


