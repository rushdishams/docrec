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
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A runnable jar file that takes directory name as input and predicts its files' ESG document type.
 * The prediction is based on an SVM model generated from the tf of the text.  
 * 
 * CHANGE:
 * Instead of sending one file, the program now reads all the files in a directory.
 * @author Rushdi Shams
 * @version 0.6 October 21 2015
 *
 */

public class FeatureGeneration {
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
		StringBuilder strBuilder = new StringBuilder();
		//		String cleanText = "";
		for (String str : content){
			//			cleanText += str + " ";
			strBuilder.append(str + " ");
		}
		String cleanText = strBuilder.toString();
		cleanText = cleanText.replaceAll("\r", " ").replaceAll("\n", " ").replaceAll("\"", "").replaceAll("\'", "");
//		System.out.println(cleanText);
		return cleanText;
	}

	public static void classify(String cleanedText, Classifier smo, String fileName){

		Attribute docText = new Attribute("doc-text", (FastVector) null);
		
		FastVector fvClassVal = new FastVector(7);
		fvClassVal.addElement("AR");
		fvClassVal.addElement("CSR");
		fvClassVal.addElement("CC");
		fvClassVal.addElement("COC");
		fvClassVal.addElement("MISC");
		fvClassVal.addElement("POLICY");
		fvClassVal.addElement("NOISE");
		Attribute classAttribute = new Attribute("doc_class", fvClassVal);

		FastVector fvWekaAttributes = new FastVector(2);
		fvWekaAttributes.addElement(docText);
		fvWekaAttributes.addElement(classAttribute);

		Instances isTrainingSet = new Instances("Relation", fvWekaAttributes, 1);
		isTrainingSet.setClassIndex(1);

		Instance iExample = new Instance(2);
		iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), "\"" + cleanedText + "\"");
		iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), "CSR");
		
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
		Classifier smo = (Classifier)new SMO();
		String[] options = null;
		try {
			options = weka.core.Utils.splitOptions("-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -M -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.NormalizedPolyKernel -C 250007 -E 2.0\"");
		} catch (Exception e1) {
			System.out.println("Options could not be created");
		}
		try {
			smo.setOptions(options);
		} catch (Exception e1) {
			System.out.println("Set options did not work");
		}
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
