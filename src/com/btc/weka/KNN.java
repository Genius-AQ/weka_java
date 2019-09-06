package com.btc.weka;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class KNN {
	
	public static void main(String[] args) throws Exception {
		
		//Input data
		BufferedReader inputData = readInputFile("iris.arff");		
		Instances data = new Instances(inputData);
		data.setClassIndex(data.numAttributes() - 1);
		
		//Training
		IBk ibk = new IBk();
		ibk.setKNN(3);
		ibk.buildClassifier(data);
		
		//Evaluate		
//		Evaluation eval = new Evaluation(data);
//		eval.evaluateModel(ibk, data);
//		System.out.println(eval.toSummaryString());
//	    System.out.println(eval.toClassDetailsString());
//		System.out.println(eval.toMatrixString());
		
		//Predict
		Instance instance = data.instance(120);
		int result = (int) ibk.classifyInstance(instance);
		System.out.println("Result: " + result);
	}
	
	public static BufferedReader readInputFile(String fileName) {
		BufferedReader bf = null;		
		try {
			 bf = new BufferedReader(new FileReader(fileName));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		return bf;
	}

}
