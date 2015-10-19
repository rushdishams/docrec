package com.sustainalytics.taggeneration;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.mcavallo.opencloud.Cloud;
import org.mcavallo.opencloud.Tag;
import org.mcavallo.opencloud.filters.LengthFilter;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A runnable jar file that takes file name as input and predicts its ESG document type
 * The prediction is based on a Naive Bayes model generated from keywords attached
 * to documents.  
 * @author Rushdi Shams
 * @version 0.3 October 16 2015
 *
 */

public class FeatureGeneration {
	public static void main(String[] args){
		String text = "";
		try {
			text = FileUtils.readFileToString(new File(args[0]));
		} catch (IOException e) {
			System.out.println("Error reading input file. Exiting.");
			System.exit(1);
		}

		String tagFeatures = getTags(text);

		Classifier nb = loadModel();

		//		System.out.println("File " + args[0] + "---->");
		classify(tagFeatures, nb);

	}

	public static void classify(String tagFeatures, Classifier nb){
		Attribute attribute = new Attribute("text", (FastVector) null);
		FastVector fvClassVal = new FastVector(7);
		fvClassVal.addElement("AR");
		fvClassVal.addElement("CSR");
		fvClassVal.addElement("CC");
		fvClassVal.addElement("COC");
		fvClassVal.addElement("MISC");
		fvClassVal.addElement("POLICY");
		fvClassVal.addElement("NOISE");
		Attribute classAttribute = new Attribute("class", fvClassVal);

		FastVector fvWekaAttributes = new FastVector(2);
		fvWekaAttributes.addElement(attribute);
		fvWekaAttributes.addElement(classAttribute);

		Instances isTrainingSet = new Instances("Relation", fvWekaAttributes, 1);
		isTrainingSet.setClassIndex(1);

		Instance iExample = new Instance(2);
		iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), tagFeatures);
		iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), "CSR");

		isTrainingSet.add(iExample);

		Instances labeled = new Instances(isTrainingSet);
        
		double clsLabel = 0;
		double[] predictionDistribution = null;
		try {
			clsLabel = nb.classifyInstance(isTrainingSet.instance(0));
			predictionDistribution = nb.distributionForInstance(isTrainingSet.instance(0));
		} catch (Exception e) {
			System.out.println("Unable to classify item\n");
		}
		labeled.instance(0).setClassValue(clsLabel);
		
		System.out.println(labeled.classAttribute().value((int) clsLabel));

		double predictionProbability = predictionDistribution[(int) clsLabel];

		System.out.println(predictionProbability );

	}

	public static Classifier loadModel(){
		Classifier nb = (Classifier)new NaiveBayes();
		try {
			nb = (Classifier) weka.core.SerializationHelper.read("model/nb-esgkeyword-model.model");
		} catch (Exception e) {
			System.out.println("Model file not found. Exiting.");
			System.exit(1);
		}
		return nb;
	}

	public static String getTags(String text){
		Cloud cloud = new Cloud();
		LengthFilter lengthFilter = new LengthFilter(10, 20);
		cloud.addInputFilter(lengthFilter);
		cloud.addText(text);
		cloud.setMaxTagsToDisplay(20);

		String weka = "\"";
		for (Tag tag : cloud.tags(new Tag.ScoreComparatorDesc())){
			weka += tag.getName() + " ";
		}
		weka += "\"";
		return weka;

	}

}