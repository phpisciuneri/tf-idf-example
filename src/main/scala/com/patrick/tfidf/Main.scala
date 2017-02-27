package com.patrick.tfidf

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.feature.{ HashingTF, IDF, Tokenizer }

object Main extends App {

  val conf = new SparkConf()
    .setAppName( "TF-IDF Example")
    .setMaster( "local[4]" )

  val sc = new SparkContext( conf )
  val sqlContext = new SQLContext( sc )

  println( "Hello" )

  val sentenceData = sqlContext.createDataFrame( Seq(
    ( 0, "new york times" ),
    ( 1, "new york post" ),
    ( 2, "los angeles times" )
  ) ).toDF( "label", "documents" )


  val tokenizer = new Tokenizer().setInputCol( "documents" ).setOutputCol( "words" )
  val wordsData = tokenizer.transform( sentenceData )

  val hashingTF = new HashingTF().setInputCol( "words" ).setOutputCol( "rawFeatures" )
  val featurizedData = hashingTF.transform( wordsData )

  val idf = new IDF().setInputCol( "rawFeatures" ).setOutputCol( "features" )
  val idfModel = idf.fit( featurizedData )
  val rescaledData = idfModel.transform( featurizedData )

  sc.stop()

}