package com.amazonaws.pa2.spark;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

//TODO  :change to mlib  -http://spark.apache.org/docs/latest/mllib-linear-methods.html#classification
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.rdd.RDD;

//""fixed acidity"",""volatile acidity"",""citric acid"",""residual sugar"",""chlorides"",""free sulfur dioxide"",""total sulfur dioxide""
//,""density"",""pH"",""sulphates"",""alcohol"",""quality""

//8.9	0.22	0.48	1.8	0.077	29	60	
// 0.9968	3.39	0.53	9.4	6
public class SparkML {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.print("Hello");
		
		SparkSession spark = SparkSession
			.builder().config("spark.testing.memory", 2147480000)
		    .master("local")
		    .appName("Programming Assignment 2: Spark")
		    .getOrCreate();
		
		StructType schema = new StructType()
		    .add("fixed acidity", "double")
		    .add("volatile acidity", "double")
		    .add("citric acid", "double")
		    .add("residual sugar", "double")
			.add("chlorides", "double")
		    .add("free sulfur dioxide", "double")
		    .add("total sulfur dioxide", "double")
		    .add("density", "double")
			.add("pH", "double")
			.add("sulphates", "double")
		    .add("alcohol", "double")
			.add("quality", "double");
		
		Dataset<Row> df = spark.read()
			    .option("mode", "DROPMALFORMED")
//			    .option("delimiter", ";")
			    .option("header", "true")
			    .schema(schema)
//			    .csv("hdfs://path/input.csv");
			    .csv("file:///Users/lopathara/Documents/Data/Shiva/MS/sem3/CloudComputing/ProgrammingAssignment2/ConsolidatedDataset.csv");
//		SparkContext sc = spark.sparkContext();
//		sc.textFile$default$2("file:///Users/lopathara/Documents/Data/Shiva/MS/sem3/CloudComputing/ProgrammingAssignment2/TrainingDataset.csv");
//		sc.textFile("file:///Users/lopathara/Documents/Data/Shiva/MS/sem3/CloudComputing/ProgrammingAssignment2/TrainingDataset.csv",100);
//		df.show();
//		Row header = df.first();
//		System.out.print("Header : "+header);


//		
//		Dataset<Row>[] splits = df.randomSplit(new double[]{0.7, 0.3}); 
//		splits[0].show(); // training
//		splits[1].show(); // test
//		
		/**
		 *********************** ML starts here **************************
		 */
		
		/**
		 * Vector Assembler - features
		 */
		VectorAssembler va = new VectorAssembler()
				.setInputCols(new String[] 
						{"fixed acidity","volatile acidity","citric acid",
								"residual sugar","chlorides","free sulfur dioxide",
								"total sulfur dioxide","density","pH","sulphates","alcohol"})
				.setOutputCol("features");   // features - MLib needs it
		Dataset<Row> vectorized_df = va.transform(df);
//		vectorized_df.show();

		/**
		 * String Indexer - label
		 */
		StringIndexer labelIndexer = new StringIndexer().setInputCol("quality").setOutputCol("label");

		Dataset<Row> vectorized_label_df =  labelIndexer.fit(vectorized_df).transform(vectorized_df);
		vectorized_label_df.show();
		
		/**
		 * Data Set Parsing Goes here
		 * Training (70) and Test Data set (30)
		 */
		Dataset<Row>[] splits = vectorized_label_df.randomSplit(new double[]{0.7, 0.3}); // with some seed
		
		Dataset<Row> trainingData = splits[0]; // training
		Dataset<Row> testData = splits[1]; // training
		
		System.out.println("Training Data Count :::::"+trainingData.count());
		System.out.println("Testing Data Count :::::"+testData.count());
		
		
		
		/**
		 * 
		 * LOGISTIC  REGRESSION - TRAIN-VALIDATION SPLIT
		 * 
		 * 
		 */
		
		LogisticRegression lr = new LogisticRegression();

		// We use a ParamGridBuilder to construct a grid of parameters to search over.
		// TrainValidationSplit will try all combinations of values and determine best model using
		// the evaluator.
		ParamMap[] paramGrid = new ParamGridBuilder()
		  .addGrid(lr.regParam(), new double[] {0.1, 0.01})
		  .addGrid(lr.maxIter(), new int [] {10, 20})
		  .addGrid(lr.fitIntercept())
		  .addGrid(lr.elasticNetParam(), new double[] {0.0, 0.5, 1.0})
		  .build();

		// In this case the estimator is simply the linear regression.
		// A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
		TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
		  .setEstimator(lr)
		  .setEvaluator(new RegressionEvaluator())
		  .setEstimatorParamMaps(paramGrid)
		  .setTrainRatio(0.8)  // 80% for training and the remaining 20% for validation
		  .setParallelism(2);  // Evaluate up to 2 parameter settings in parallel

		// Run train validation split, and choose the best set of parameters.
		TrainValidationSplitModel model = trainValidationSplit.fit(trainingData);

		// Make predictions on test data. model is the model with combination of parameters
		// that performed best.
		Dataset<Row> predictionsDF = model.transform(testData);
		predictionsDF.select("features", "label", "prediction").show();
		
		
		/***
		 * 
		 * Evaluation
		 * 
		 * 
		 */
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
		evaluator.setLabelCol("label");
		
		double accuracy = evaluator.evaluate(predictionsDF);
		System.out.println("Accuracy : "+accuracy);
		
		
		/**
		 * Confusion Matrix
		 */
		MulticlassMetrics metrics = new MulticlassMetrics(predictionsDF.select("prediction", "label"));
		System.out.println(metrics.confusionMatrix());
		
		System.out.println(metrics.fMeasure(1.0)); //F score
		
		/****
		 * 
		 * 
		 * Decision Tree Classifier
		 * 
		 * 
		 * 
		 * 
		 */
		
//		StringIndexerModel slabelIndexer = new StringIndexer()
//				  .setInputCol("label")
//				  .setOutputCol("indexedLabel")
//				  .fit(trainingData);
//
//				// Automatically identify categorical features, and index them.
//				VectorIndexerModel featureIndexer = new VectorIndexer()
//				  .setInputCol("features")
//				  .setOutputCol("indexedFeatures")
//				  .setMaxCategories(11) // features with > 4 distinct values are treated as continuous.
//				  .fit(trainingData);
//
////				// Split the data into training and test sets (30% held out for testing).
////				Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
////				Dataset<Row> trainingData = splits[0];
////				Dataset<Row> testData = splits[1];
//
//				// Train a DecisionTree model.
//				DecisionTreeClassifier dt = new DecisionTreeClassifier()
//				  .setLabelCol("indexedLabel")
//				  .setFeaturesCol("indexedFeatures");
//
//				// Convert indexed labels back to original labels.
//				IndexToString labelConverter = new IndexToString()
//				  .setInputCol("prediction")
//				  .setOutputCol("predictedLabel")
//				  .setLabels(slabelIndexer.labelsArray()[0]);
//
//				// Chain indexers and tree in a Pipeline.
//				Pipeline pipeline = new Pipeline()
//				  .setStages(new PipelineStage[]{slabelIndexer, featureIndexer, dt, labelConverter});
//
//				// Train model. This also runs the indexers.
//				PipelineModel model = pipeline.fit(trainingData);
//
//				// Make predictions.
//				Dataset<Row> predictions = model.transform(testData);
//
//				// Select example rows to display.
//				predictions.select("predictedLabel", "label", "features").show(5);
//
//				// Select (prediction, true label) and compute test error.
//				MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
//				  .setLabelCol("indexedLabel")
//				  .setPredictionCol("prediction")
//				  .setMetricName("accuracy");
//				double accuracy = evaluator.evaluate(predictions);
//				System.out.println("Test Error = " + (1.0 - accuracy));
//
//				DecisionTreeClassificationModel treeModel =
//				  (DecisionTreeClassificationModel) (model.stages()[2]);
//				System.out.println("Learned classification tree model:\n" + treeModel.toDebugString());
	/****
	 * 
	 * Multi Layer Perceptron
	 * 
	 * 
	 */



//		// specify layers for the neural network:
//		// input layer of size 11 (features), two intermediate of size 5 and 4
//		// and output of size 10 (classes)
//		int[] layers = new int[] {11, 5, 4, 6};
//
//		// create the trainer and set its parameters
//		MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
//		  .setLayers(layers)
//		  .setBlockSize(128)
//		  .setSeed(1234L)
//		  .setMaxIter(500);
//
//		// train the model
//		MultilayerPerceptronClassificationModel model = trainer.fit(trainingData);
//
//		// compute accuracy on the test set
//		Dataset<Row> result = model.transform(testData);
//		Dataset<Row> predictionAndLabels = result.select("prediction", "label");
//		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
//		  .setMetricName("accuracy");
//
//		System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));
//		
//		
		
		/**
		 * 
		 * 
		 * Gradient Boosted Tree Regression
		 * 
		 * 
		 */
		

//		// Automatically identify categorical features, and index them.
//		// Set maxCategories so features with > 4 distinct values are treated as continuous.
//		VectorIndexerModel featureIndexer = new VectorIndexer()
//		  .setInputCol("features")
//		  .setOutputCol("indexedFeatures")
//		  .setMaxCategories(4)
//		  .fit(trainingData);
//
//		// Split the data into training and test sets (30% held out for testing).
////		Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
////		Dataset<Row> trainingData = splits[0];
////		Dataset<Row> testData = splits[1];
//
//		// Train a GBT model.
//		GBTRegressor gbt = new GBTRegressor()
//		  .setLabelCol("label")
//		  .setFeaturesCol("indexedFeatures")
//		  .setMaxIter(10);
//
//		// Chain indexer and GBT in a Pipeline.
//		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {featureIndexer, gbt});
//
//		// Train model. This also runs the indexer.
//		PipelineModel model = pipeline.fit(trainingData);
//
//		// Make predictions.
//		Dataset<Row> predictions = model.transform(testData);
//
//		// Select example rows to display.
//		predictions.select("prediction", "label", "features").show(5);
//
//		// Select (prediction, true label) and compute test error.
//		RegressionEvaluator evaluator = new RegressionEvaluator()
//		  .setLabelCol("label")
//		  .setPredictionCol("prediction")
//		  .setMetricName("rmse");
//		double rmse = evaluator.evaluate(predictions);
//		System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
//		
//		
//		
//		MulticlassClassificationEvaluator evaluatorMulticlass = new MulticlassClassificationEvaluator()
//		  .setMetricName("accuracy");
//
//		System.out.println("Test set accuracy = " + evaluatorMulticlass.evaluate(predictions.select("prediction", "label", "features")));
//		
//
//		GBTRegressionModel gbtModel = (GBTRegressionModel)(model.stages()[1]);
//		System.out.println("Learned regression GBT model:\n" + gbtModel.toDebugString());

		
		
		
		
		/**
		 * Estimator
		 */
//		// Create a LogisticRegression instance. This instance is an Estimator.
//		LogisticRegression lrEstimator = new LogisticRegression();
//		// Print out the parameters, documentation, and any default values.
//		System.out.println("LogisticRegression parameters:\n" + lrEstimator.explainParams() + "\n");
//
//		// We may set parameters using setter methods.
//		lrEstimator.setMaxIter(500).setElasticNetParam(0.8).setRegParam(0.01);    // very important hyper parameters
//		
//		
//		/**
//		 * Transformer - Training Data (trainingData)
//		 */
//		// Learn a LogisticRegression model. This uses the parameters stored in lr.
//		LogisticRegressionModel transformedModel = lrEstimator.fit(trainingData);
//		// Since model1 is a Model (i.e., a Transformer produced by an Estimator),
//		// we can view the parameters it used during fit().
//		// This prints the parameter (name: value) pairs, where names are unique IDs for this
//		// LogisticRegression instance.
////		transformedModel.setThreshold(0.5);
//		System.out.println("Model 1 was fit using parameters: " + transformedModel.parent().extractParamMap());
//		
//		/**
//		 * Test Data (testData)
//		 */
//		Dataset<Row> predictionsDF =  transformedModel.transform(testData);
//		System.out.print(transformedModel.summary().accuracy());
//		
//		
//
////		transformedModel.predict(predictionsDF.col("features"));
////		predictionsDF.show();
//		predictionsDF.select("prediction", "label", "features").show();
//		
//		predictionsDF.select("prediction", "label", "features").groupBy("prediction").count().show();
//		
//		
//		/**
//		 * Evaluation
//		 */
//		//JavaRDD<LabeledPoint> data  = MLUtils.
//		
//		// Compute raw scores on the test set.
////		JavaPairRDD<Object, Object> predictionAndLabels = testData.mapToPair(p ->
////		  new Tuple2<>(model.predict(p.features()), p.label()));
////
////		// Get evaluation metrics.
////		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
////
////		// Confusion matrix
////		Matrix confusion = metrics.confusionMatrix();
////		System.out.println("Confusion matrix: \n" + confusion);
//		
//		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
//		evaluator.setLabelCol("label");
//
//		
//		/**
//		 * Accuracy
//		 */
//		double accuracy = evaluator.evaluate(predictionsDF);
//		System.out.println("Accuracy : "+accuracy);
//		
//		/**
//		 * Confusion Matrix
//		 */
//		MulticlassMetrics metrics = new MulticlassMetrics(predictionsDF.select("prediction", "label"));
//		System.out.println(metrics.confusionMatrix());
//		
//		System.out.println(metrics.fMeasure(1.0)); //F score
//

	}
	


}
