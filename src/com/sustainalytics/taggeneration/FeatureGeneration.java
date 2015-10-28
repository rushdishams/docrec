package com.sustainalytics.taggeneration;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

import com.boundary.sentence.TextContent;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A runnable jar file that takes directory name as input and predicts its
 * files' ESG document type. The prediction is based on an NB model generated
 * from the tf-idf of the text.
 * 
 * CHANGE: Object oriented structure
 * 
 * @author Rushdi Shams
 * @version 0.9 October 28 2015
 *
 */

public class FeatureGeneration {
	// ----------------------------------------------------------------------------------------
	// Instance Variables
	// ----------------------------------------------------------------------------------------

	private int length; // document length
	private Classifier classifier;
	private File folder;
	private String modelPath, text, cleanedText, currentFileName;
	private File[] listOfFiles;

	// Instance variables for Weka
	private Instances labeled, isTrainingSet;
	Instance iExample;
	private double clsLabel = 0, predictionProbability = 0;
	private Attribute docText, docLength, classAttribute;
	private FastVector fvClassVal, fvWekaAttributes;

	/**
	 * Constructor. Sets the folder that contains files to be classified and the
	 * path to model.
	 * 
	 * @param directory
	 *            that contains files to be classified
	 * @param modelPath
	 *            path of the saved model
	 */
	public FeatureGeneration(String directory, String modelPath) {
		this.folder = new File(directory);
		this.listOfFiles = folder.listFiles();
		this.modelPath = modelPath;
	}

	/**
	 * returns list of files in the directory
	 * 
	 * @return a File Array
	 */
	public File[] getListOfFiles() {
		return this.listOfFiles;
	}

	/**
	 * Method to set the text of the file
	 * 
	 * @param text
	 *            is a String
	 */
	public void setText(String text) {
		this.text = text;
	}

	/**
	 * Method used to set the file name that is currently getting processed
	 * 
	 * @param currentFileName
	 *            is a String
	 */
	public void setCurrentFileName(String currentFileName) {
		this.currentFileName = currentFileName;
	}

	/**
	 * toString() method to display file name, its class and model's probability
	 * for the class
	 */
	public String toString() {
		return currentFileName + "\t" + labeled.classAttribute().value((int) clsLabel) + "\t" + predictionProbability;
	}

	/**
	 * Method to clean the text of the text file. It uses lingpipe to get the
	 * sentence boundaries and replaces newlines with whitespaces and replaces
	 * quotes with empty string.
	 */
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

	/**
	 * Method to create a mock dataset from test file.
	 */
	public void initializeDataset() {
		// Text and length attributes
		this.docText = new Attribute("doc-text", (FastVector) null);
		this.docLength = new Attribute("doc-length");

		// Class attribute
		this.fvClassVal = new FastVector(2);
		this.fvClassVal.addElement("MISC");
		this.fvClassVal.addElement("NOISE");
		this.classAttribute = new Attribute("doc_class", fvClassVal);

		// All the attributes
		this.fvWekaAttributes = new FastVector(3);
		this.fvWekaAttributes.addElement(docText);
		this.fvWekaAttributes.addElement(docLength);
		this.fvWekaAttributes.addElement(classAttribute);

		// Creating a dataset with name Relation and with 1 instance in it
		isTrainingSet = new Instances("Relation", fvWekaAttributes, 1);
		// What is the index for class attribute?
		isTrainingSet.setClassIndex(2);
		// The instance will be having three attributes
		iExample = new Instance(3);
	}

	/**
	 * Method to classify a test example
	 */
	public void classify() {
		// Setting the attribute values for the instance
		this.iExample.setValue((Attribute) fvWekaAttributes.elementAt(0), "\"" + this.cleanedText + "\"");
		this.iExample.setValue((Attribute) fvWekaAttributes.elementAt(1), this.length);
		this.iExample.setValue((Attribute) fvWekaAttributes.elementAt(2), "MISC"); // Aribtrary
		// class
		// is
		// provided.

		// putting the instance to the mock dataset
		this.isTrainingSet.add(iExample);

		// will contain the labeled instance
		this.labeled = new Instances(isTrainingSet);

		double[] predictionDistribution = null;
		try {
			this.clsLabel = this.classifier.classifyInstance(isTrainingSet.instance(0));
			predictionDistribution = this.classifier.distributionForInstance(isTrainingSet.instance(0));
		} catch (Exception e) {
			System.out.println("Unable to classify item\n");
		}
		this.labeled.instance(0).setClassValue(clsLabel);
		this.predictionProbability = predictionDistribution[(int) clsLabel];
	}

	/**
	 * Method to load model and setting it to a classifier
	 */
	public void loadModel() {

		// setting the classifier--->
		try {
			this.classifier = (Classifier) weka.core.SerializationHelper.read(this.modelPath);
		} catch (Exception e) {
			System.out.println("Model file not found. Exiting.");
			System.exit(1);
		}

	}

	// ------------------------------------------------------------------------------------
	// Driver class
	// ------------------------------------------------------------------------------------
	public static void main(String[] args) {
		Instant start = Instant.now();

		if (!new File(args[0]).isDirectory()) {
			System.out.println("Input must be a directory.");
			System.exit(1);
		}

		FeatureGeneration test = new FeatureGeneration(args[0], args[1]);
		test.loadModel();
		File[] listOfFiles = test.getListOfFiles();
		// For each file in the directory ---->
		for (int i = 0; i < listOfFiles.length; i++) {
			// If we find a PDF, then try to find out its text version, if
			// found, then we will proceed to classify --->
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
				System.gc();// For less memory consumption
			} // <--- text version was found
		} // <--- iterating over all files is done

		Instant end = Instant.now();
		System.out.println("Completion time: " + Duration.between(start, end));
	}// end driver method
}// end class
