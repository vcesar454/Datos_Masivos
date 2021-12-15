import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{StringIndexer, IndexToString}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.MulticlassMetrics

def DecisionTreeA(N_iterations : Int): Unit = {

    var i = 1;
    //Creating and reading the dataset
    val sparkSession1 = SparkSession.builder.getOrCreate()
    val bankDF = sparkSession1.read.format("csv").option("inferSchema", "true").option("delimiter", ";").option("header", "true").load("dataset/bank.csv")

    //Assembeling the features column
    val feature_assembler = new VectorAssembler().setInputCols(Array("balance", "day", "duration", "campaign", "pdays", "previous")).setOutputCol("features")
    val features = feature_assembler.transform(bankDF)

    //Indexing the output label
    val labelsIndexed = new StringIndexer().setInputCol("y").setOutputCol("y_label")
    val dataIndexed = labelsIndexed.fit(features).transform(features)

    //Indexing all y labels and features
    val labelIndexer = new StringIndexer().setInputCol("y_label").setOutputCol("indexed_y_label").fit(dataIndexed)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)

    //Spliting the dataset into two parts
    val Array(trainingData, testingData) = dataIndexed.randomSplit(Array(0.7, 0.3))

    //mapItera is used to record the iteration number and accuracy
    val mapItera = scala.collection.mutable.Map(0 -> 0.0)

    while(i <= N_iterations){
        //Measuring Execution time per run
        val t1 = System.nanoTime
        
        //Creating, the Classifier and indicationg the labels column and features column; reverting the indexed labels back to strings; lastly setting
        //up the Pipeline
        val dt = new DecisionTreeClassifier().setLabelCol("indexed_y_label").setFeaturesCol("indexedFeatures")
        val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
        val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

        //Model
        val model = pipeline.fit(trainingData)

        //Predincting Results using the testing data
        val predictions = model.transform(testingData)

        //Evaluating the accuracy of the model
        val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexed_y_label").setPredictionCol("prediction").setMetricName("accuracy")

        val accuracy = evaluator.evaluate(predictions)
        
        println(s"Accuracy = $accuracy" + "\n")
        mapItera += (i -> accuracy)
        i = i + 1
    }
    println("Printing the map for GBT Classifier:\n")
    mapItera.keys.foreach{ e => 
        print("Iteration = " + e + " ")
        println("Accuracy = " + mapItera(e)) 
    }
}