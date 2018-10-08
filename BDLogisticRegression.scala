import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.functions._


object BDLogisticRegression
{
  def main(args: Array[String]) {
    if (args.length == 0) {
      println("i need two  parameters ")
    }
    val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("BDLogisticRegression"))
    //val sc = new SparkContext(new SparkConf().setAppName("BDLogisticRegression"))

    val spark = SparkSession
      .builder()
      .appName("BDLogisticRegression")
      .getOrCreate()

    import spark.implicits._

    var df = spark.read.json(args(0))

    df=df.withColumnRenamed("overall", "label")

    var filteredData = df.select("reviewerID","reviewerName","reviewText","label")

    filteredData = filteredData.withColumn("reviewText",regexp_replace($"reviewText","[^a-zA-Z0-9\\s+]",""))

    var logisticRegressionData = filteredData.withColumn("label",filteredData("label") - 1.0)
    logisticRegressionData =logisticRegressionData.orderBy(asc("reviewerID"))


    val stringTokenizer = new Tokenizer()
      .setInputCol("reviewText")
      .setOutputCol("words")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(stringTokenizer.getOutputCol)
      .setOutputCol("filtered_words")

    val countVectorizer = new CountVectorizer()
      .setInputCol("filtered_words")
      .setOutputCol("features")

    val Array(trainingData, testData) =logisticRegressionData.randomSplit(Array(0.8, 0.2), seed= 1234L)

    val classificationModel = new LogisticRegression().setMaxIter(10)

    val pipeline = new Pipeline().setStages(Array(stringTokenizer, stopWordsRemover, countVectorizer, classificationModel))

    val paramGridModelLR = new ParamGridBuilder()
      .addGrid(classificationModel.regParam, Array(0.3,0.01))
      .addGrid(classificationModel.elasticNetParam, Array(0.8, 0.4))
      .build()

    val multiclassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val crossValidatorModel_3 = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(multiclassClassificationEvaluator)
      .setEstimatorParamMaps(paramGridModelLR)
      .setNumFolds(3)

    val cvLr = crossValidatorModel_3.fit(trainingData)

    val outputPredictions = cvLr.transform(testData)

    val res = outputPredictions.select("reviewerID","reviewerName","prediction")

    var negReviewCount = res.filter($"prediction" <=2).groupBy($"reviewerID").count().withColumnRenamed("count","neg_review_count")

    var totalReviews = res.groupBy($"reviewerID").count().withColumnRenamed("count","total_review_count")

    var res3 = negReviewCount.join(totalReviews,Seq("reviewerID"),"inner")

    var finalresult = res3.withColumn("percentage",(col("neg_review_count") / col("total_review_count")* 100))

    var negreviewers = finalresult.filter($"percentage">60)

    var output = "Negative Reviewers list with more than 60% negative reviews :\n "
    negreviewers.rdd.collect().foreach { case (row) => {
      output += row(0) + " with " + row(3) +"%\n"
    }}

    multiclassClassificationEvaluator.setMetricName("accuracy")
    val accuracy = multiclassClassificationEvaluator.evaluate(outputPredictions)

    multiclassClassificationEvaluator.setMetricName("weightedPrecision")
    val precision = multiclassClassificationEvaluator.evaluate(outputPredictions)

    multiclassClassificationEvaluator.setMetricName("weightedRecall")
    val recall = multiclassClassificationEvaluator.evaluate(outputPredictions)

    multiclassClassificationEvaluator.setMetricName("f1")
    val f1 = multiclassClassificationEvaluator.evaluate(outputPredictions)

    val outputString = "Logistic Regression"+ "\nAccuracy: " + accuracy + "\nWeighted Precision: " + precision + "\nWeighted Recall: " + recall +"\n F Measure: "+ f1 + "\n" + output

    sc.parallelize(List(outputString)).repartition(1).saveAsTextFile(args(1))

  }

}
