package com.amazonaws.pa2.spark;

import java.io.IOException;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.optimization.LogisticGradient;
import org.apache.spark.mllib.optimization.SquaredL2Updater;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

//""fixed acidity"",""volatile acidity"",""citric acid"",""residual sugar"",""chlorides"",""free sulfur dioxide"",""total sulfur dioxide""
//,""density"",""pH"",""sulphates"",""alcohol"",""quality""

//8.9	0.22	0.48	1.8	0.077	29	60	
// 0.9968	3.39	0.53	9.4	6
public class MLTrainSparkMlib {

	public static void main(String[] args) throws IOException {
		System.out.println("Traning ML in Spark with multi-ec2 cluster starts.....");
		String current = new java.io.File( "." ).getCanonicalPath();
        System.out.println("Current dir:"+current);
        String currentDir = System.getProperty("user.dir");
        System.out.println("Current dir using System:" +currentDir);
			
        /**
         * Spark Context
         * 
         */
        
        SparkSession spark = SparkSession
		    .builder().config("spark.testing.memory", 2147480000)
		    	.master("local") //TODO : remove when deploying
		    .appName("Programming Assignment 2: Spark")
		    .getOrCreate();
		
        /**
         * Construct schema to Spark and Download Data from csv via spark context
         * 
         */
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
			    .option("delimiter", ";")
			    .option("header", "true")
			    .schema(schema)
//			    .csv("file:///data/TrainingDataset.csv"); // TODO : change while deploying to environment  //TODO : Important
//				.csv("hdfs:///data/TrainingDataset.csv"); // TODO : change while deploying to environment  //TODO : Important
			    .csv("file:///Users/lopathara/Documents/Data/Shiva/MS/sem3/CloudComputing/ProgrammingAssignment2/TrainingDataset.csv");
		
		df.show();
		
		
//		
		/**
		 *********************** ML starts here **************************
		 */
		
		/**
		 * Data Preparation and feature extraction
		 */
		
		/**
		 * Vector Assembler - construct 'features'
		 */
		VectorAssembler va = new VectorAssembler()
				.setInputCols(new String[] 
						{"fixed acidity","volatile acidity","citric acid",
								"residual sugar","chlorides","free sulfur dioxide",
								"total sulfur dioxide","density","pH","sulphates","alcohol"})
				.setOutputCol("features");   // features - MLib needs it

		
		
		Dataset<Row> vectorized_df = va.transform(df);

		/**
		 * String Indexer - construct 'label'
		 */

		StringIndexer labelIndexer = new StringIndexer().setInputCol("quality").setOutputCol("label");
		Dataset<Row> vectorized_label_df =  labelIndexer.fit(vectorized_df).transform(vectorized_df);

		
		/**
		 * Data Set Parsing Goes here
		 * Training (70) and Test Data set (30)
		 */
		Dataset<Row>[] splits = vectorized_label_df.randomSplit(new double[]{0.9, 0.1}); // TODO : change threshold before submitting
		
		Dataset<Row> trainingData = splits[0]; // training
		Dataset<Row> testData = splits[1]; // training
		
		/**
		 * ****** conversion to RDD for MLIB
		 */
	
		JavaRDD<Row> trainingDataRDD =    trainingData.javaRDD();
		JavaRDD<Row> testDataRDD =    testData.javaRDD();
		
		
		JavaRDD<LabeledPoint> trainingDataLabeledPoints = trainingDataRDD.map(new Function <Row, LabeledPoint>()
		{
		             @Override
		            public LabeledPoint call(Row line) throws Exception {
		            	org.apache.spark.mllib.linalg.Vector vect = Vectors.fromML(line.getAs("features"));
		                LabeledPoint labeledPoint = new LabeledPoint(line.getAs("label"), vect);
		                return labeledPoint;
		            }
				

        });
		JavaRDD<LabeledPoint> testDataLabeledPoints = testDataRDD.map(new Function <Row, LabeledPoint>()
		{
		             @Override
		            public LabeledPoint call(Row line) throws Exception {
		            	org.apache.spark.mllib.linalg.Vector vect = Vectors.fromML(line.getAs("features"));
		                LabeledPoint labeledPoint = new LabeledPoint(line.getAs("label"), vect);
		                return labeledPoint;
		            }
				

        });
		System.out.println("Training Data Count :::::"+trainingData.count());
		System.out.println("Testing Data Count :::::"+testData.count());
		
		
		/**
		 * 
		 *  Train ML with Logistic Regression Model, LBFGS optimizer and Logistic Gradient
		 */
		
		LogisticRegressionWithLBFGS transformedModelLBFGS = new LogisticRegressionWithLBFGS();
		
		/**
		 * 
		 * Set Optimizer parameters
		 */
		
		transformedModelLBFGS.optimizer()
		.setRegParam(0.3)
		.setGradient(new LogisticGradient())
		.setUpdater(new SquaredL2Updater())
		.setNumCorrections(10)
		.setConvergenceTol(0.00001)
		.setNumIterations(500); // TODO : set other hyper params
		
		//multi-classe wine-quality score 1-10 
		transformedModelLBFGS.setNumClasses(10);
		
//		Gradient Updates
//		transformedModelLBFGS.optimizer().setGradient(new Gradient() {
//			
//			@Override
//			public double compute(Vector data, double label, Vector weights, Vector cumGradient) {
//				// TODO Auto-generated method stub
//				return 20;
//				
//				var w = Vector.zeros(d)
//						for (i <- 1 to numIterations) {
//						val gradient = points.map { p =>
//						(1 / (1 + exp(-p.y * w.dot(p.x)) - 1) * p.y * p.x
//						).reduce(_ + _)
//						w -= alpha * gradient }
//				
//			}
//		})
		

		
		/**
		 *  Training (RUN)
		 *  
		 */
		LogisticRegressionModel transformedModel = transformedModelLBFGS.run(trainingDataLabeledPoints.rdd()); //convert to RDD<LabeledPoint>
		
		
		/**
		 * Test Data Prediction (testData)
		 */
		
		JavaPairRDD<Object, Object> predictionAndLabels = testDataLabeledPoints.mapToPair(p ->
		  new Tuple2<>(transformedModel.predict(p.features()), p.label()));
		
		
		/**
		 * F1 score and Evaluation
		 */
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		double accuracy = metrics.accuracy();
		
		System.out.println("*************************************************");
		System.out.println("Accuracy = " + accuracy);
		
		// F Score by threshold

		System.out.println("F1 Score:: " + metrics.weightedFMeasure());
		System.out.println("*************************************************");
		
		/***
		 * Confusion matrix 
		 * 
		 */
		
		Matrix confusion = metrics.confusionMatrix();
		System.out.println("Confusion matrix: \n" + confusion);

		
		/**
		 * Save and load model 
		 * 
		 */
		// get SparkCOntext from Spark Session -spark.sparkContext()
		
		transformedModel.save(spark.sparkContext(), "target/tmp/javaLogisticRegressionWithLBFGSModel");
		//Load the model
		LogisticRegressionModel sameModel = LogisticRegressionModel.load(spark.sparkContext(),
		  "target/tmp/javaLogisticRegressionWithLBFGSModel");



	}
	


}
