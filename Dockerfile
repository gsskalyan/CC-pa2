FROM openjdk:8
WORKDIR .
ADD ShivaPA2-1.0.0.jar ShivaPA2-1.0.0.jar
ADD target target
EXPOSE 8080
ENTRYPOINT ["java", "-cp","ShivaPA2-1.0.0.jar","com.amazonaws.pa2.spark.SparkWinePrediction"]