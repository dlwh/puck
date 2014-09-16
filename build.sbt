import AssemblyKeys._ // put this at the top of the file

organization := "org.scalanlp"

name := "puck"

version := "0.1"

scalaVersion := "2.10.3"

libraryDependencies ++= Seq(
  "junit" % "junit" % "4.5" % "test",
  "org.scalanlp" %% "epic" % "0.2",
  "org.scalatest" %% "scalatest" % "2.0.M5b" % "test",
  "com.nativelibs4java" % "javacl" % "1.0-SNAPSHOT"
)

libraryDependencies += "com.jsuereth" %% "scala-arm" % "1.3"

fork := true

javaOptions ++= Seq("-Xmx12g", "-Xrunhprof:cpu=samples,depth=12")
//javaOptions ++= Seq("-Xmx12g")




resolvers ++= Seq(
  "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
)

assemblySettings

mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
  {
    case PathList("META-INF", xs @ _*) =>
      (xs map {_.toLowerCase}) match {
        case ("manifest.mf" :: Nil) | ("index.list" :: Nil) | ("dependencies" :: Nil) =>
          MergeStrategy.discard
        case ps @ (x :: xs) if ps.last.endsWith(".sf") || ps.last.endsWith(".dsa") =>
          MergeStrategy.discard
        case "plexus" :: xs =>
          MergeStrategy.discard
        case "services" :: xs =>
          MergeStrategy.filterDistinctLines
        case ("spring.schemas" :: Nil) | ("spring.handlers" :: Nil) =>
          MergeStrategy.filterDistinctLines
        case _ => MergeStrategy.first
      }
      case x => MergeStrategy.first
  }
}

excludedJars in assembly <<= (fullClasspath in assembly) map { cp => 
  cp filter {_.data.getName.contains("dx-")}
}


testOptions in Test += Tests.Argument("-oDF")

