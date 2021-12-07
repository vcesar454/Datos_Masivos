//Import a simple Spark session
import org.apache.spark.sql.SparkSession
//Import the Kmeans library for the clustering algorithm
import org.apache.spark.ml.clustering.KMeans
//import VectorAssembler and Vector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

//Create a simple Spark Session
val session = SparkSession.builder().getOrCreate()

//Load the Wholesale Customers Data dataset
val data = session.read.option("header", "true").option("inferSchema", "true").format("csv").load("CSV/Wholesale customers data.csv")

// Select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data
var feature_data = data.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

//Create a new Vector Assembler object for the feature columns as an input set, remembering that there are no labels
val vector_assembler = (new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features"))

//Use the assembler object to transform feature_data
val data_kmeans = vector_assembler.transform(feature_data)

// Create a Kmeans model with K = 3
val model = new KMeans().setK(3).setSeed(1L)

//Evaluate the groups using Within Set Sum of Squared Errors WSSSE and print the  centroids.
val result = model.fit(data_kmeans)

import org.apache.spark.ml.evaluation.ClusteringEvaluator
val WSSSE = result.computeCost(data_kmeans)
println(s"Within set sum of squared errors = $WSSSE")

//Print the centroids
println("Cluster Centers:")
result.clusterCenters.foreach(println)
