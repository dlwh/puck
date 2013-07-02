organization := "org.scalanlp"

name := "puck"

version := "0.1-SNAPSHOT"

resolvers += ScalaToolsSnapshots

scalaOrganization := "org.scala-lang.virtualized"

scalaVersion := "2.10.1"

scalacOptions += "-Yvirtualize"

libraryDependencies ++= Seq(
  "junit" % "junit" % "4.5" % "test",
  "org.scalanlp" %% "breeze-core" % "0.3-SNAPSHOT",
  "org.scalanlp" %% "breeze-math" % "0.3-SNAPSHOT",
  "org.scalanlp" %% "trochee" % "0.1-SNAPSHOT",
  "org.scalanlp" %% "epic" % "0.1-SNAPSHOT",
  "org.scalatest" %% "scalatest" % "2.0.M5b" % "test"
)

fork := true
