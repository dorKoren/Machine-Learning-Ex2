package main;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import main.DecisionTree.eImpurityMode;
import main.DecisionTree.ePruningMode;
import weka.core.Instances;

public class MainHW2 {
	
	public static final double[] P_VALUES = {1, 0.75, 0.5, 0.25, 0.05, 0.005};
	public static final double[][] TABLE_OF_CHISQUARED_PROBABILITIES =
			{
					
			//	  1    0.75      0.5       0.25       0.05      0.005	
				{ 0,   0.102,   0.455 ,    1.323 ,   3.841 ,    7.879  },  // deg 1         
				{ 0,   0.575,   1.386 ,    2.773 ,   5.991 ,    10.597 },  // deg 2         
				{ 0,   1.213,   2.366 ,    4.108 ,   7.815 ,    12.838 },  // deg 3         
				{ 0,   1.923,   3.357 ,    5.385 ,   9.488 ,    14.860 },  // deg 4         
				{ 0,   2.675,   4.351 ,    6.626 ,   11.070,    16.750 },  // deg 5         
				{ 0,   3.455,   5.348 ,    7.841 ,   12.592,    18.548 },  // deg 6         
				{ 0,   4.255,   6.346 ,    9.037 ,   14.067,    20.278 },  // deg 7         
				{ 0,   5.071,   7.344 ,    10.219,   15.507,    21.955 },  // deg 8         
				{ 0,   5.899,   8.343 ,    11.389,   16.919,    23.589 },  // deg 9        
				{ 0,   6.737,   9.342 ,    12.549,   18.307,    25.188 },  // deg 10        
				{ 0,   7.584,   10.341,    13.701,   19.675,    26.757 },  // deg 11        
				{ 0,   8.438,   11.340,    14.845,   21.026,    28.300 }   // deg 12        
			};


	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
	
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	
	public static void main(String[] args) throws Exception {
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");
		
		DecisionTree entropyDT = new DecisionTree(eImpurityMode.Entropy, ePruningMode.None);
		DecisionTree giniDT = new DecisionTree(eImpurityMode.Gini, ePruningMode.None); 

		/*** Start of part 1 ****/
		
		// (1.a) 
		// Build the decision tree with Entropy as impurity measure 
		// using the training set.
		entropyDT.buildClassifier(trainingCancer);
	
		// Calculate the average error on the validation set, based on the
		// tree that built with respect to the Entropy. 
		double entropyValidationError = entropyDT.calcAvgError(validationCancer); 
		
		System.out.println("Validation error using Entropy: " + entropyValidationError);
		
		// (1.b) 
		// Build the decision tree with Gini as impurity measure 
		// using the training set.
		giniDT.buildClassifier(trainingCancer);
	
		// Calculate the average error on the validation set, based on the
		// tree that built with respect to the Gini.
		double giniValidationError = giniDT.calcAvgError(validationCancer);	
		
		System.out.println("Validation error using Gini: " + giniValidationError +"\n");
				
		// (1.c)
		// Choose the impurity measure that gave you the lowest validation
		// error. Use this impuruty measure for the rest of the task.
		
		eImpurityMode bestImpurity;
		
		bestImpurity = entropyValidationError < giniValidationError ? eImpurityMode.Entropy : eImpurityMode.Gini;
		
		/*** End of part 1 ****/
		
		
		
		/*** Start of part 2 ****/
		
		/* For each p-value cutoff value do the following: */
		
		System.out.println("--------------------------------------------------------");
		
		
		// (2.a)
		// Construct a tree and prune it according to the current cutoff value.
		double bestValidationErrorAtPValue = Double.MAX_VALUE;
		int bestPValueIndex = -1;
		   
		for (int i = 0; i < P_VALUES.length; i++) {
						
			double p_value = P_VALUES[i];
			System.out.println("Decision Tree with p_value of: " + p_value);
			
			DecisionTree dt = new DecisionTree(bestImpurity, ePruningMode.Chi, TABLE_OF_CHISQUARED_PROBABILITIES, i);
			
			dt.buildClassifier(trainingCancer);
			
			
			// (2.b)
			// Calculate training error.
			double trainingError = dt.calcAvgError(trainingCancer);
			System.out.println("The trein error of the decision tree is: " + trainingError);
			
			// (2.c)
			// Calculate the tree average & max heights according to the 
			// validation set as described above, with respect to the following 
			// p- values cutoffs: {1 (no pruning), 0.75, 0.5, 0.25, 0.05, 0.005}.
			
			
			// Set the 'm_CountHeighs' field to be 0.
			dt.setTotalHeights(0);
			dt.setCountHeight(0);
			
			double validatinError = dt.calcAvgError(validationCancer); // Calculate validation error.
			int validationMaxHeight = dt.getCountHeight();
			double validationAverageHeight = (double) dt.getTotalHeighs() / validationCancer.numInstances() ;
		
			System.out.println("Max height on validation data: " + validationMaxHeight);
			
			System.out.println("Average height on validation data: " + validationAverageHeight); 
		
			System.out.println("The validation error of the decision tree is: " + validatinError);
			
			// Update the best p_value index and the best validation 
			// error if necessary.
			if (validatinError < bestValidationErrorAtPValue) {
				bestValidationErrorAtPValue = validatinError;
				bestPValueIndex = i;
			}

			System.out.println("--------------------------------------------------------");
		}
		
		/*** End of part 2 ****/
		
		
		/*** Start of part 3 ****/
				
		/* Select the cutoff that resulted in the best validation error: */
		
		// (3.a)
		// Calculate the test error for the tree corresponding to this configuration.
		System.out.println("Best validation error at p_value = " + bestValidationErrorAtPValue);  
		
		// Set the p - value of the decision tree to be the best one 
		// that we found above.
		DecisionTree bestDT = new DecisionTree(bestImpurity, ePruningMode.Chi, TABLE_OF_CHISQUARED_PROBABILITIES, bestPValueIndex);
		
		// build decision tree according to the the best p - value that we 
		// found above, then calc average error of testing data.
		// NOTE: we set already set the pruning mode field to "chi mode".
		bestDT.buildClassifier(trainingCancer);
		
		
		double testError = bestDT.calcAvgError(testingCancer);
		System.out.println("Test error with best tree: " + testError);
		
		// (3.b)
		// Print the corresponding tree to the console as described above.
		System.out.println("Representation of the best tree by 'if statment': " + "\n" );
		bestDT.printTree();                                           

		
		/*** End of part 3 ****/




		

		

		
		

		

		
		
	}
}