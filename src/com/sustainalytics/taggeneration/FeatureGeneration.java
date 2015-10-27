package com.sustainalytics.taggeneration;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

import com.boundary.sentence.TextContent;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A runnable jar file that takes directory name as input and predicts its
 * files' ESG document type. The prediction is based on an SVM model generated
 * from the tf of the text.
 * 
 * CHANGE: Two Class rather than three. It considers document length as a
 * feature too.
 * 
 * @author Rushdi Shams
 * @version 0.9 October 27 2015
 *
 */

public class FeatureGeneration {
	private int length;
	private Classifier classifier;
	private File folder;
	private String modelPath, text, cleanedText, currentFileName;
	private File[] listOfFiles;
	private Instances labeled;
	private double clsLabel = 0, predictionProbability = 0;
	private Attribute docText, docLength, classAttribute;

	private FastVector fvClassVal, fvWekaAttributes;


	private Instances isTrainingSet;

	private Instance iExample;

	public FeatureGeneration(String directory, String modelPath) {
		this.folder = new File(directory);
		this.listOfFiles = folder.listFiles();
		this.modelPath = modelPath;

	}

	public File[] getListOfFiles() {
		return this.listOfFiles;
	}

	public void setText(String text) {
		this.text = text;
	}

	public void setCurrentFileName(String currentFileName) {
		this.currentFileName = currentFileName;

	}

	public String toString() {
		return currentFileName + "\t" + labeled.classAttribute().value((int) clsLabel) + "\t" + predictionProbability;
	}

	

	public void cleanText() {
		TextContent t = new TextContent(); // creating TextContent object
		t.setText(this.text);
		t.setSentenceBoundary();
		String[] content = t.getSentence();
		this.length = content.length;
		StringBuilder strBuilder = new StringBuilder();
		for (String str : content) {
			strBuilder.append(str.trim() + " ");
		}
		String cleanText = strBuilder.toString();
		cleanText = cleanText.replaceAll("\r", " ").replaceAll("\n", " ").replaceAll("\"", "").replaceAll("\'", "");
		this.cleanedText = cleanText;
	}
	
	public void initializeDataset(){
		this.docText = new Attribute("doc-text", (FastVector) null);
		this.docLength = new Attribute("doc-length");

		this.fvClassVal = new FastVector(2);
		this.fvClassVal.addElement("MISC");
		this.fvClassVal.addElement("NOISE");
		this.classAttribute = new Attribute("doc_class", fvClassVal);

		this.fvWekaAttributes = new FastVector(3);
		this.fvWekaAttributes.addElement(docText);
		this.fvWekaAttributes.addElement(docLength);
		this.fvWekaAttributes.addElement(classAttribute);

		isTrainingSet = new Instances("Relation", fvWekaAttributes, 1);
		isTrainingSet.setClassIndex(2);

		iExample = new Instance(3);
	}

	public void classify() {

		
		this.iExample.setValue((Attribute) fvWekaAttributes.elementAt(0), "\"" + this.cleanedText + "\"");
		this.iExample.setValue((Attribute) fvWekaAttributes.elementAt(1), this.length);
		this.iExample.setValue((Attribute) fvWekaAttributes.elementAt(2), "MISC");

		this.isTrainingSet.add(iExample);

		this.labeled = new Instances(isTrainingSet);
		
		double[] predictionDistribution = null;
		try {
			this.clsLabel = this.classifier.classifyInstance(isTrainingSet.instance(0));
			predictionDistribution = classifier.distributionForInstance(isTrainingSet.instance(0));
		} catch (Exception e) {
			System.out.println("Unable to classify item\n");
		}
		this.labeled.instance(0).setClassValue(clsLabel);

		this.predictionProbability = predictionDistribution[(int) clsLabel];

	}

	public void loadModel() {
		this.classifier = (Classifier) new NaiveBayes();
		try {
			this.classifier = (Classifier) weka.core.SerializationHelper.read(this.modelPath);
		} catch (Exception e) {
			System.out.println("Model file not found. Exiting.");
			System.exit(1);
		}
	}
	
	public static void main(String[] args) {

		Instant start = Instant.now();

		if (!new File(args[0]).isDirectory()) {
			System.out.println("Input must be a directory.");
			System.exit(1);
		}

		FeatureGeneration test = new FeatureGeneration(args[0], args[1]);
		test.loadModel();
		File[] listOfFiles = test.getListOfFiles();
		for (int i = 0; i < listOfFiles.length; i++) {
			if (FilenameUtils.getExtension(listOfFiles[i].getAbsolutePath()).equalsIgnoreCase("pdf")) {
				try {
					test.setText(FileUtils.readFileToString(
							new File(FilenameUtils.removeExtension(listOfFiles[i].getAbsolutePath()) + ".txt")));
				} catch (IOException e) {
					System.out.println(listOfFiles[i].getName()
							+ "--No associated .txt file for the PDF. Moving to the next file.");
					continue;
				}

				test.setCurrentFileName(listOfFiles[i].getName());
				test.cleanText();
				test.initializeDataset();
				test.classify();
				System.out.println(test.toString());
				System.gc();
			}
		}

		Instant end = Instant.now();
		System.out.println("Completion time: " + Duration.between(start, end));
	}
}
