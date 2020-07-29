
GIT HUB [SOURCE] : https://github.com/gsskalyan/CC-pa2

DOCKER HUB [IMAGES] : https://hub.docker.com/r/ss4477/pa2/tags

IMPORTANT : Command to PULL / RUN Docker Hub Image:

RUN PREDICTION APP WITH DOCKER DESKTOP LOCAL
	-	docker run -v "$(pwd)":/data ss4477/pa2:v1

RUN PREDICTION APP WITH DOCKER in EC2
	-	sudo docker run -v "$(pwd)":/data ss4477/pa2:v1

MUST READ (before running above docker run command):
	1.	Make sure to keep the input file named "TestDataset.csv" in your present working directory from where you are running above docker command
	2.	No need to modify anything in above command.Your current working directory will be mounted to docker container with name /data
	


JAVA FILES

	1.	MLTrainSparkMlib.java
		⁃	Java Program that uses ‘Logistic Regression’ to train in multiple instances
		⁃	Saves the trained model in to target/tmp/
	2.	SparkWinePrediction.java
		⁃	Prediction application the loads the trained model
		⁃	Reads ‘TestDataset.csv’ from present working directory
		⁃	Outputs F1 score
	3.	SparkML.java
		⁃	Additionally, tried training the Model with other ML algorithms like Random Forest,Decision Tree (commented out before checking in to Git hub though)


Docker Commands [Local]:

	1.	Docker command used Locally
	2.	docker build -t shiva—pa2-docker .
	3.	docker image ls
	4.	docker container ls
	5.	docker tag e464d57427f9 ss4477/pa2:v1
	6.	docker run -v "$(pwd)":/data ss4477/pa2:v1

Docker Commands [Install in AMI]:
	1.	sudo yum update -y
	2.	sudo yum install docker -y
	3.	sudo service docker start
	4.	docker —version


CREATE SPARK CLUSTER AND TRAIN ML MODEL:

Spark cluster [Create EMR Spark cluster ,Train ML model using multiple EC2]
	1.	Cluster Name : Test Spark [Followed:https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-launch.html].Refer detailed screen prints attached in a separate document - CS643_PA2_CreateCLusterAndML_ParallelTraining_Proof.pdf
	2.	Connect to one master(cluster): 
		⁃	ssh -i ~/shiva-key.pem hadoop@ec2-34-204-196-140.compute-1.amazonaws.com
		Connect to three slaves(cluster): 
		-	ssh -i "shiva-key.pem" hadoop@ec2-35-171-164-174.compute-1.amazonaws.com
		-	ssh -i "shiva-key.pem" hadoop@ec2-3-237-34-52.compute-1.amazonaws.com
		-	ssh -i "shiva-key.pem" hadoop@ec2-3-236-123-47.compute-1.amazonaws.com
	3.	Download the Jar (ML Training - Parallely in multiple EC2 instances) from my S3 Bucket (not a requirement)- 
		⁃	aws s3 cp s3://shivapa2/ShivaPA2-1.0.0.jar .
		⁃	aws s3 cp s3://shivapa2/TrainingDataset.csv .	[in both master and slave]
		⁃	scp -i "shiva-key.pem" /Users/lopathara/eclipse-workspace/CC-PA2-Spark/target/ShivaPA2-1.0.0.jar  hadoop@ec2-54-147-160-139.compute-1.amazonaws.com:/home/hadoop/ShivaPA2-1.0.0.jar
	5.	spark-submit --class com.amazonaws.pa2.spark.MLTrainSparkMlib ShivaPA2-1.0.0.jar file:///data/TrainingDataset.csv 
	6.	Download the saved Model
		⁃	scp -i ~/shiva-key.pem -r hadoop@ec2-54-147-160-139.compute-1.amazonaws.com:TrainingDataset.csv .
		⁃	scp -i ~/shiva-key.pem -r hadoop@ec2-54-147-160-139.compute-1.amazonaws.com:target/tmp/javaLogisticRegressionWithLBFGSModel .

RUN WINE PREDICTION MODEL ON EC2:

Creating an EC2 instance [For running Saved ML and run docker.]

1.	Using AWS educate starter account [I used student account]
	⁃	Login in to https://www.awseducate.com/student/s/awssite
	⁃	Navigate to Vocareum https://labs.vocareum.com/main/main.php?m=editor&nav=1&asnid=14334&stepid=14335
	⁃	Click AWS Console - https://console.aws.amazon.com/console/home?region=us-east-1#
	⁃	Choose ‘Launch a Virtual Machine’ in AWS services
	⁃	Follow below 7 steps
	⁃	Step1 - Choose an Amazon Machine Image (AMI) [Amazon Linux2 AMI]
	⁃	Step2 - Choose Instance type (t2.micro) - with 1 EBS
	⁃	Step3 - Configure Instance Details - number of instances as 2
	⁃	Step4 - Add Storage (8 or 16 GB)
	⁃	Step5 - Add Tags
	⁃	Step6 - Configure Security Group (Example name:launch-wizard-4)
	⁃	Select MYIP from source
	⁃	Ports opened SSH,HTTP,HTTPS
	⁃	Step7 - Review and Launch
2.	Select an existing  key pair or create a new key-pair
	⁃	Download .pem key “shiva-key.pem” and save it under home directory locally.
	⁃	This is required to SSH to EC2 instance from local ‘Terminal’ (Mac)
3.	View EC2 Dashboard - https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Home:
	⁃	Running Instances - Select - Connect 

Connect to an EC2 instance
1.	Open Command prompt(Putty) or Terminal(Mac).
	⁃	Enter following command (EC2-A)
	⁃	ssh -i "shiva-key.pem" ec2-user@ec2-100-25-138-193.compute-1.amazonaws.com
	⁃	Use same command to connect to (EC2-B),but by changing the corresponding Public DNS (IPv4)
	⁃	ssh -i "shiva-key.pem" ec2-user@ec2-18-209-158-193.compute-1.amazonaws.com
	⁃ Note - Public DNS (IPv4) : ec2-user@ec2-52-91-119-71 will change for new instance or every time you start/stop)

RUN PREDICTION APP WITHOUT DOCKER in EC2 [Instance ID: i-07afb4e2c45374411] - Refer detailed screen prints attached in a separate document - CS643_PA2_WinePredict_WithAndWithoutDockerEC2.pdf
	⁃	scp -i "shiva-key.pem" ~/.aws/credentials ec2-user@ec2-100-25-134-232.compute-1.amazonaws.com
	-	scp -i "shiva-key.pem" -r /Users/lopathara/Downloads/jre-8u251-linux-x64.tar ec2-user@ec2-100-25-134-232.compute-1.amazonaws.com:/home/ec2-user/jre-8u251-linux-x64.tar	
	-	ssh -i "shiva-key.pem" ec2-user@ec2-100-25-134-232.compute-1.amazonaws.com
	-	To Download Artifacts
	-	aws s3 cp s3://shivapa2/ShivaPA2-1.0.0.jar .
	⁃	aws s3 cp s3://shivapa2/TestDataset.csv .
	⁃	aws s3 cp s3://shivapa2/target . --recursive
	-	To Install Java :
	⁃	cd /home/ec2-user/java
	⁃	To extract tar 
	⁃	[ec2-user@ip-172-31-51-52 java]$ tar xvf jre-8u251-linux-x64.tar
	⁃	Extracted output will have jre1.8.0_251
	⁃	Set JAVA to PATH
	⁃	[ec2-user@ip-172-31-51-52 bin]$ echo $PATH
	⁃	/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/ec2-user/.local/bin:/home/ec2-user/bin
	⁃	[ec2-user@ip-172-31-51-52 bin]$ export JAVA_HOME=/home/ec2-user/java/jre1.8.0_251/
	⁃	[ec2-user@ip-172-31-51-52 bin]$ PATH=$PATH:$JAVA_HOME/bin
	⁃	[ec2-user@ip-172-31-51-52 bin]$ export PATH
	⁃	[ec2-user@ip-172-31-51-52 bin]$ java -version
	⁃	java version "1.8.0_251"
	-	RUN WITHOUT DOCKER IN EC2
	-	java -cp ShivaPA2-1.0.0.jar com.amazonaws.pa2.spark.SparkWinePrediction


RUN PREDICTION APP WITH DOCKER in EC2 [Instance ID: i-07afb4e2c45374411] - Refer detailed screen prints here at - CS643_PA1_DOCKER_Setup_Proof.pdf
	1.	sudo yum update -y
	2.	sudo yum install docker -y
	3.	sudo service docker start
	4.	docker —version
	5.	sudo docker run -v "$(pwd)":/data ss4477/pa2:v1



CONFIGURE AND INSTALL [Local instruction]

1.	AWS config and credentials 
	⁃	cd /home/ec2-user
	⁃	mkdir .aws
	⁃	vi credentials
	⁃	Now go to Vocareum - Account Details - AWS CLI (click Show).[Note that was-cli credentials is valid only for 3 hours]
	⁃	Get ‘’aws_access_key_id”, “aws_secret_access_key” and “aws_session_token” along with its values and save it to credentials under /home/ec2-user/.aws/credentails (or copy from local, if you have it)
	⁃	EC2-A
	⁃	scp -i "shiva-key.pem" /Users/lopathara/.aws/credentials ec2-user@ec2-100-25-134-232.compute-1.amazonaws.com:/home/ec2-user/.aws/


GIT COMMANDS:

	git init
	git add README.md
	git commit -m "first commit"
	git remote add origin https://github.com/gsskalyan/CC-pa2.git
	git push -u origin master
