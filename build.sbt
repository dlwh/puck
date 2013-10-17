organization := "org.scalanlp"

name := "puck"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.3"

libraryDependencies ++= Seq(
  "junit" % "junit" % "4.5" % "test",
  "org.scalanlp" %% "breeze" % "0.5",
  "org.scalanlp" %% "epic" % "0.1-SNAPSHOT",
  "org.scalatest" %% "scalatest" % "2.0.M5b" % "test",
  "com.nativelibs4java" % "javacl" % "1.0.0-RC3"
)

fork := true

javaOptions ++= Seq("-Xmx6g")




resolvers ++= Seq(
  "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
)
