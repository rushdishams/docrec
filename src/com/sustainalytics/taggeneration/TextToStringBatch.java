package com.sustainalytics.taggeneration;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;

import com.boundary.sentence.TextContent;

/**
 * Class to generate an ARFF file from a directory of documents. The ARFF file has two attributes:
 * cleaned text and class.
 * @author Rushdi Shams
 * @version 0.1.0 October 20 2015
 *
 */

public class TextToStringBatch {
	public static void main(String[] args){
		File folder = new File(args[0]);
		File[] listOfFiles = folder.listFiles();
		String wekaOutput = "@relation kw-based-classification\n\n@attribute text String\n@attribute class{AR, CSR, CC, COC, POLICY, MISC, NOISE}\n\n@data\n\n";
		System.out.println(listOfFiles.length);
		for (int i = 0; i < listOfFiles.length; i++){
			System.out.println(i);
			String text = "";
			try {
				text = FileUtils.readFileToString(listOfFiles[i].getAbsoluteFile());
			} catch (IOException e) {
				System.out.println("Error reading input file");
			}
			
			TextContent t = new TextContent(); //creating TextContent object
			t.setText(text);
			t.setSentenceBoundary();
			String[] content = t.getSentence();
			StringBuilder strBuilder = new StringBuilder();
//			String cleanText = "";
			for (String str : content){
//				cleanText += str + " ";
				strBuilder.append(str + " ");
			}
			String cleanText = strBuilder.toString();
			wekaOutput += "\"" + cleanText.replaceAll("\r", " ").replaceAll("\n", " ").replaceAll("\"", "").replaceAll("\'", "") + "\"";
			wekaOutput += args[1] + "\n";
		}
		try {
			FileUtils.write(new File(folder.getPath() + "/" + "wekaoutput.arff"), wekaOutput);
		} catch (IOException e) {
			System.out.println("Error in writing output");
		}
	}

}
