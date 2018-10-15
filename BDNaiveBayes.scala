import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._


object BDNaiveBayes
{
  def main(args: Array[String]) {
    if (args.length == 0) {
      println("i need two  parameters ")
    }
    val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("NaiveBayes"))
    //val sc = new SparkContext(new SparkConf().setAppName("NaiveBayes"))

//    val spark = SparkSession
//      .builder()
//      .appName("NaiveBayes")
//      .config("spark.master", "local[*]")
//      .getOrCreate()


    val spark = SparkSession
      .builder()
      .appName("NaiveBayes")
      .getOrCreate()

    import spark.implicits._

    var df = spark.read.json(args(0))

    df=df.withColumnRenamed("overall", "label")

    var filteredData = df.select("reviewerID","reviewerName","reviewText","label")

    filteredData = filteredData.withColumn("reviewText",regexp_replace($"reviewText","[^a-zA-Z0-9\\s+]",""))

    //changed because Naive Bayes outputs classes indexed from 0 - 4 instead of input classes 1-5
    var naiveBayesData = filteredData.withColumn("label",filteredData("label") - 1.0)
    naiveBayesData = naiveBayesData.orderBy(asc("reviewerID"))

    val stringTokenizer = new Tokenizer()
      .setInputCol("reviewText")
      .setOutputCol("words")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(stringTokenizer.getOutputCol)
      .setOutputCol("filtered_words")

    val countVectorizer = new CountVectorizer()
      .setInputCol("filtered_words")
      .setOutputCol("features")

    val Array(trainingData, testData) = naiveBayesData.randomSplit(Array(0.8, 0.2), seed= 1234L)

    // Train a NaiveBayes model.
    val classificationModel = new NaiveBayes()

    val pipeline = new Pipeline().setStages(Array(stringTokenizer, stopWordsRemover, countVectorizer, classificationModel))

    val paramGridModelNB = new ParamGridBuilder()
      .addGrid(classificationModel.smoothing, Array(0.5, 1.0, 1.5, 2.0))
      .build()

    val multiclassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val crossValidatorModel_3 = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(multiclassClassificationEvaluator)
      .setEstimatorParamMaps(paramGridModelNB)
      .setNumFolds(3)

    val cvNb = crossValidatorModel_3.fit(trainingData)

    val outputPredictions = cvNb.transform(testData)


    multiclassClassificationEvaluator.setMetricName("accuracy")
    val accuracy = multiclassClassificationEvaluator.evaluate(outputPredictions)

    multiclassClassificationEvaluator.setMetricName("weightedPrecision")
    val precision = multiclassClassificationEvaluator.evaluate(outputPredictions)

    multiclassClassificationEvaluator.setMetricName("weightedRecall")
    val recall = multiclassClassificationEvaluator.evaluate(outputPredictions)

    multiclassClassificationEvaluator.setMetricName("f1")
    val f1 = multiclassClassificationEvaluator.evaluate(outputPredictions)

    //accuracy
    val outputString = "Naive Bayes"+ "\nAccuracy: " + accuracy + "\nWeighted Precision: " + precision + "\nWeighted Recall: " + recall +"\n F Measure: "+ f1

    val res = outputPredictions.select("reviewerID","reviewerName","prediction")

    var negReviewCount = res.filter($"prediction" <=2).groupBy($"reviewerID").count().withColumnRenamed("count","neg_review_count")

    var totalReviews = res.groupBy($"reviewerID").count().withColumnRenamed("count","total_review_count")

    var res3 = negReviewCount.join(totalReviews,Seq("reviewerID"),"inner")

    var finalresult = res3.withColumn("percentage",(col("neg_review_count") / col("total_review_count")* 100))

    var negreviewers = finalresult.filter($"percentage">60)

    var output = outputString +  "\n Negative Reviewers list with more than 60% negative reviews :\n "
    negreviewers.rdd.collect().foreach { case (row) => {
      output += row(0) + " with " + row(3) +"%\n"
    }}

    sc.parallelize(List(output)).repartition(1).saveAsTextFile(args(1))

  }

}
