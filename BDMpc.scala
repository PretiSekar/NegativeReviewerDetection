import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.sql.functions.{col, lit, when}

object BDMpc
{
  def main(args: Array[String]) {
    if (args.length == 0) {
      println("i need two  parameters ")
    }

    val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("BDMpc"))
    //val sc = new SparkContext(new SparkConf().setAppName("BDMpc"))

    val spark = SparkSession
      .builder()
      .appName("BDMpc")
      .getOrCreate()

    import spark.implicits._

    var df = spark.read.json(args(0))

    df=df.withColumnRenamed("overall", "label")

    var filteredData = df.select("reviewerID","reviewerName","reviewText","label")

    filteredData = filteredData.withColumn("reviewText",regexp_replace($"reviewText","[^a-zA-Z0-9\\s+]",""))

    val stringTokenizer = new Tokenizer()
      .setInputCol("reviewText")
      .setOutputCol("words")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(stringTokenizer.getOutputCol)
      .setOutputCol("filtered_words")

    val wordVectorizer = new Word2Vec()
      .setInputCol("filtered_words")
      .setOutputCol("result")
      .setVectorSize(20)
      .setMinCount(0)

    val vectorIndexer = new VectorIndexer()
      .setInputCol("result")
      .setOutputCol("features")
      .setMaxCategories(20)


    val Array(trainingData, testData) = filteredData.randomSplit(Array(0.8, 0.2), seed= 1234L)

    val layers = Array[Int](20,6)
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setLabelCol("label")
      .setBlockSize(128)

    val pipeline = new Pipeline().setStages(Array(stringTokenizer, stopWordsRemover, wordVectorizer, vectorIndexer, trainer))

    val paramGridModelMPC = new ParamGridBuilder()
      .addGrid(trainer.maxIter, Array(10,15))
      .build()

    val multiclassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val crossValidationModel_3 = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGridModelMPC)
      .setEvaluator(multiclassClassificationEvaluator)
      .setNumFolds(3)

    val cvMpc = crossValidationModel_3.fit(trainingData)

    val outputPredictions = cvMpc.transform(testData)

    multiclassClassificationEvaluator.setMetricName("accuracy")
    val accuracy = multiclassClassificationEvaluator.evaluate(outputPredictions)

    multiclassClassificationEvaluator.setMetricName("weightedPrecision")
    val precision = multiclassClassificationEvaluator.evaluate(outputPredictions)

    multiclassClassificationEvaluator.setMetricName("weightedRecall")
    val recall = multiclassClassificationEvaluator.evaluate(outputPredictions)

    multiclassClassificationEvaluator.setMetricName("f1")
    val f1 = multiclassClassificationEvaluator.evaluate(outputPredictions)

    val outputString = "Multilayer Perceptron Classifier"+ "\nAccuracy: " + accuracy + "\nWeighted Precision: " + precision + "\nWeighted Recall: " + recall +"\n F Measure: "+ f1
	
	val res = outputPredictions.select("reviewerID","reviewerName","prediction")

	var negReviewCount = res.filter($"prediction" <3).groupBy($"reviewerID").count().withColumnRenamed("count","neg_review_count")
	
	var totalReviews = res.groupBy($"reviewerID").count().withColumnRenamed("count","total_review_count")
	
	var res3 = negReviewCount.join(totalReviews,Seq("reviewerID"),"inner")
	
	var finalresult = res3.withColumn("percentage",(col("neg_review_count") / col("total_review_count")* 100))

    var negreviewers = finalresult.filter($"percentage">60)

    var output = outputString +  "\n Negative Reviewers list with more than 60% negative reviews :\n "
    negreviewers.rdd.collect().foreach { case (row) => {
      output += row(0) + " with " + row(3) +"%\n"
    }}
    sc.parallelize(List(output)).saveAsTextFile(args(1))

  }

}
