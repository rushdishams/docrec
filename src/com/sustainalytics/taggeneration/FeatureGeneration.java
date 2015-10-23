package com.sustainalytics.taggeneration;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.mcavallo.opencloud.Cloud;
import org.mcavallo.opencloud.Tag;
import org.mcavallo.opencloud.filters.LengthFilter;

import com.boundary.sentence.TextContent;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A runnable jar file that takes directory name as input and predicts its files' ESG document type.
 * The prediction is based on an SVM model generated from the tf of the text.  
 * 
 * CHANGE:
 * Two Class rather than three. It considers document length as a feature too.
 * @author Rushdi Shams
 * @version 0.8 October 23 2015
 *
 */

public class FeatureGeneration {
	private static int length;
	public static void main(String[] args){

		Instant start = Instant.now();

		File folder = new File(args[0]);
		String modelPath = args[1];
		if(!folder.isDirectory()){
			System.out.println("Input must be a directory.");
			System.exit(1);
		}
		File[] listOfFiles = folder.listFiles();
		for (int i = 0; i < listOfFiles.length; i++){
			if(FilenameUtils.getExtension(listOfFiles[i].getAbsolutePath()).equalsIgnoreCase("pdf")){

				String text = "";
				try {
					text = FileUtils.readFileToString(new File(FilenameUtils.removeExtension(listOfFiles[i].getAbsolutePath()) + ".txt"));
				} catch (IOException e) {
					System.out.println(listOfFiles[i].getName() + "--No associated .txt file for the PDF. Moving to the next file.");
					continue;
				}

				String cleanedText = cleanText(text);

				Classifier smo = loadModel(modelPath);

				//		System.out.println("File " + args[0] + "---->");
				classify(cleanedText, smo, listOfFiles[i].getName());
			}

		}
		Instant end = Instant.now();
		System.out.println("Completion time: " + Duration.between(start, end));
	}

	public static String cleanText(String text){
		TextContent t = new TextContent(); //creating TextContent object
		t.setText(text);
		t.setSentenceBoundary();
		String[] content = t.getSentence();
		length = content.length;
		StringBuilder strBuilder = new StringBuilder();
		//		String cleanText = "";
		for (String str : content){
			//			cleanText += str + " ";
			strBuilder.append(str.trim() + " ");
		}
		String cleanText = strBuilder.toString();
		cleanText = cleanText.replaceAll("\r", " ").replaceAll("\n", " ").replaceAll("\"", "").replaceAll("\'", "");
//		System.out.println(cleanText);
		return cleanText;
	}

	public static void classify(String cleanedText, Classifier smo, String fileName){

		Attribute docText = new Attribute("doc-text", (FastVector) null);
		Attribute docLength = new Attribute("doc-length");
		
		FastVector fvClassVal = new FastVector(2);
		fvClassVal.addElement("MISC");
		fvClassVal.addElement("NOISE");
		Attribute classAttribute = new Attribute("doc_class", fvClassVal);

		FastVector fvWekaAttributes = new FastVector(3);
		fvWekaAttributes.addElement(docText);
		fvWekaAttributes.addElement(docLength);
		fvWekaAttributes.addElement(classAttribute);

		Instances isTrainingSet = new Instances("Relation", fvWekaAttributes, 1);
		isTrainingSet.setClassIndex(2);

		Instance iExample = new Instance(3);
		iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), "\"" + cleanedText + "\"");
		iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), length);
		iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), "MISC");
		
		isTrainingSet.add(iExample);

		Instances labeled = new Instances(isTrainingSet);

		double clsLabel = 0;
		double[] predictionDistribution = null;
		try {
			clsLabel = smo.classifyInstance(isTrainingSet.instance(0));
			predictionDistribution = smo.distributionForInstance(isTrainingSet.instance(0));
		} catch (Exception e) {
			System.out.println("Unable to classify item\n");
		}
		labeled.instance(0).setClassValue(clsLabel);


		double predictionProbability = predictionDistribution[(int) clsLabel];
		System.out.println(fileName + "\t" + labeled.classAttribute().value((int) clsLabel) + "\t" + predictionProbability);

	}

	public static Classifier loadModel(String modelPath){
		Classifier smo = (Classifier)new NaiveBayes();
//		String[] options = null;
//		try {
//			options = weka.core.Utils.splitOptions("-I 100 -K 0 -S 1");
//		} catch (Exception e1) {
//			System.out.println("Options could not be created");
//		}
//		try {
//			smo.setOptions(options);
//		} catch (Exception e1) {
//			System.out.println("Set options did not work");
//		}
		try {
			smo = (Classifier) weka.core.SerializationHelper.read(modelPath);
		} catch (Exception e) {
			System.out.println("Model file not found. Exiting.");
			System.exit(1);
		}
		return smo;
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
