package com.amazonaws.pa2.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

public class SparkWinePrediction {

	public static void main(String[] args) {
        SparkSession spark = SparkSession
		    .builder().config("spark.testing.memory", 2147480000)
		    .master("local") //TODO : remove when deploying
		    .appName("Programming Assignment 2: Spark Prediction App")
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
				.format("csv")
				.option("delimiter", ";")
			    .option("mode", "DROPMALFORMED")
			    .option("header", "true")
			    .schema(schema)
			    .csv("/data/TestDataset.csv"); // TODO : change while deploying to environment to Test name
//			    .load("file:///Users/lopathara/Documents/Data/Shiva/MS/sem3/CloudComputing/ProgrammingAssignment2/ValidationDataset.csv");
		
		df.show();
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
		
		/**
		 * String Indexer - label
		 */

		StringIndexer labelIndexer = new StringIndexer().setInputCol("quality").setOutputCol("label");

		Dataset<Row> vectorized_label_df =  labelIndexer.fit(vectorized_df).transform(vectorized_df);
		vectorized_label_df.show();
		
		/**
		 * ****** conversion for MLIB
		 */
		
		JavaRDD<Row> validationDataRDD =    vectorized_label_df.javaRDD(); 
		
	
		
		JavaRDD<LabeledPoint> validationTestDataLabeledPoints = validationDataRDD.map(new Function <Row, LabeledPoint>()
		{
		             @Override
		            public LabeledPoint call(Row line) throws Exception {
		            	 System.out.println("Label"+line.getAs("label"));
		            	 System.out.println("features"+line.getAs("features"));
		            	 org.apache.spark.mllib.linalg.Vector vect = Vectors.fromML(line.getAs("features"));  // vect
		                LabeledPoint labeledPoint = new LabeledPoint(line.getAs("label"), vect);
		                return labeledPoint;
		            }
				

        });
		
		
		/**
		 * loading  a trained Model that was already saved 
		 */
		//hdfs://ip-172-31-16-40.ec2.internal:8020/user/hadoop/target/tmp/javaLogisticRegressionWithLBFGSModel
			
		LogisticRegressionModel sameModel = LogisticRegressionModel.load(spark.sparkContext(),
				  "target/tmp/javaLogisticRegressionWithLBFGSModel"); // TODO : change the model
		
		
		/**
		 * Prediction
		 */
		
		JavaPairRDD<Object, Object> predictionAndLabels = validationTestDataLabeledPoints.mapToPair(p ->     // TODO  :Provide better name
		  new Tuple2<>(sameModel.predict(p.features()), p.label()));
		
		
		/**
		 * F1 score and Evaluation
		 */
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		
		/***
		 * Confusion matrix 
		 * 
		 */
		
		Matrix confusion = metrics.confusionMatrix();
		System.out.println("Confusion matrix: \n" + confusion);
		
		System.out.print("Prediction and labels"+predictionAndLabels);
		
		double accuracy = metrics.accuracy();
		System.out.println("*************************************************");
		System.out.println("Accuracy = " + accuracy);
		System.out.println("*************************************************");
		
		// F Score by threshold
		System.out.println("*************************************************");
		System.out.println("F1 Score:: " + metrics.weightedFMeasure());
		System.out.println("*************************************************");
		

	}

}
