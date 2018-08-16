package main;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;


class Node {
	Node[] children;     
	Node parent;         
	int attributeIndex; 
	double returnValue;
	double branch; 
	List<Double> attributeValuesBranches; 

}
	

public class DecisionTree implements Classifier {
	private Node rootNode;
	public enum ePruningMode {None, Chi};
	public enum eImpurityMode {Entropy, Gini};
	private eImpurityMode m_ImpurutyMode;
	private ePruningMode m_PruningMode; 
	private double[][] m_TableOfChiProb;
	private int m_PValueIndex;
	int m_TotalHeighs = 0;
	int m_CountHeight = 0;
	
	
	/**
	 * Constructor that create a Decision tree object.
	 * We will use this constructor for the *first* part of the assignment.
	 * @param pruningMode
	 * @param impurutyMode
	 */
	public DecisionTree (eImpurityMode impurutyMode, ePruningMode pruningMode) {
		
		m_PruningMode = pruningMode;
		m_ImpurutyMode = impurutyMode;
	}
	
	
	/**
	 * Constructor that create a Decision tree object.
	 * We will use this constructor for the *second* part of the assignment.
	 * @param impurutyMode
	 * @param pruningMode
	 * @param tableOfChiProb
	 * @param PValueIndex

	 */
	public DecisionTree (eImpurityMode impurutyMode, ePruningMode pruningMode, 
			double[][] tableOfChiProb, int PValueIndex) {
		
		m_PruningMode = pruningMode;
		m_ImpurutyMode = impurutyMode;
		m_TableOfChiProb = tableOfChiProb;
		m_PValueIndex = PValueIndex;

	}
	
	
	/* sets & gets methods */
	
	public Node getRoot() {
		return this.rootNode;
	}
	
	public eImpurityMode getImpurityMode() {
		return this.m_ImpurutyMode;
	}
	
	public int getCountHeight() {
		return this.m_CountHeight;
	}
	
	public void setCountHeight(int countHeight) {
		this.m_CountHeight = countHeight;
	}
	

	public int getTotalHeighs() {
		return this.m_TotalHeighs;
	}
	
	public void setTotalHeights(int totalHeights) {
		this.m_TotalHeighs = totalHeights;
	}
	

	public double[][] getTableOfChiProb() {
		return this.m_TableOfChiProb;
	}
	

	public int getPValueIndex() {
		return this.m_PValueIndex;
	}
	
	/* public methods */
	
	/**
	 * Build a decision tree from the training data. buildClassifier is 
	 * seperated from buildTree mehod in order to allow us to do extra 
	 * preprocessing before calling buildTree mehod or post processing after.
	 * @param arg0
	 * @throws Exception
	 */
	@Override
	public void buildClassifier(Instances trainingInstances) throws Exception {
		
     	// Build the decision tree with respect to the given impurity mode.
		this.buildTree(trainingInstances, this.getImpurityMode());	                                                                             
	}
	
	
	/**
	 * Return the classification of the instance.
	 * @param instance
	 * @return double number, 0 or 1, represent the classified class.
	 */
    @Override
	public double classifyInstance(Instance instance) {
    	Node currentNode = this.getRoot();
    	double cuurentReturningValue = currentNode.returnValue;
    	int currentAttributeIndex = currentNode.attributeIndex;
    	int counter = 0;
    	
    	// if the node has children, send the current instance 
    	// to the correspond child node.
    	while (currentNode.children != null) {
    		double instanceAttributeVal = instance.value(currentAttributeIndex);
    		
    		if (!(currentNode.attributeValuesBranches.contains(instanceAttributeVal))) {
    			break;
    			
    		} else {
    			
    			for (Node childNode : currentNode.children) {
    				if (childNode.branch == instanceAttributeVal) {
    					currentNode = childNode;
    					cuurentReturningValue = childNode.returnValue;
    					currentAttributeIndex = currentNode.attributeIndex;
    					
    	    			// Update the height of the instance. We will use it 
    	    			// for calc the max height & average height of the validation data.
    	    			counter += 1;
    				}
    			}
    		}
    	}
    	
    	// Update the height of the field.
    	this.setCountHeight(counter);

    	return cuurentReturningValue;
    }
 

    /**
     * Builds the decision tree on given data set using either a recursive 
     * or queue algorithm.
     * @param instance
     * @param impurityMode                       
     */
    public void buildTree(Instances instances, eImpurityMode impurityMode) {
    	
    	this.rootNode = buildTreeRec(null, instances, impurityMode, -1);                            
    }
    
   
	/**
     * Calculate the average error on a given instances set 
     * (could be the training, test or validation set). The average error 
     * is the total number of classification mistakes on the input instances 
     * set divided by the number of instances in the input set.
     * @param instances to operate on.
     * @return average error classification error.
     */
    public double calcAvgError(Instances instances) {
    	int errorCounter = 0;
    	int maxHeight = Integer.MIN_VALUE;
    	int totalHeights = 0;
    	
    	// Iterate through all instances, and count classification errors.
    	for (int i = 0; i < instances.numInstances(); i++) {
    		Instance instance = instances.instance(i);
    		
    		double classifyInstance = this.classifyInstance(instance);
    		
			// Sum all the height. We will use this data to calculate 
			// the validation height average error.
			totalHeights += this.getCountHeight();
				
			// Update the max height every iteration when updating 
			// the 'count_heigh' field. 
			if (this.getCountHeight() > maxHeight) {
				maxHeight = this.getCountHeight();
			}
			
			// Set the countHeight field to 0 for the next iteration.
			this.setCountHeight(0);
			
  	        // Update the number of error counter.
    		if (classifyInstance != instance.classValue()) {
    			errorCounter++;
    		}
    	}
    	
    	this.setCountHeight(maxHeight);
    	this.setTotalHeights(totalHeights);
    	
    	// Return average clasiification error. 
    	return (double) errorCounter / instances.numInstances();
    }
     
    
    /**
     * Calculates the instances entropy with respect to the their 
     * different class value.
     * @param  instances
     * @return entropy value
     */
    public double calcEntropy(Instances instances) { 
    	// Get class distribution.
    	Map<Double, Integer> classDistriburion = this.classDistribution(instances);
    	
    	// Calculate the entropy (in it's negated sign).
    	double entropy = 0;
    	
    	for (Map.Entry<Double, Integer> classEntry : classDistriburion.entrySet()) {
    		int distribution = classEntry.getValue();
    		if (distribution == 0) {
    			// Ignore non existent class values.
    			continue;
    		}
    		
    		double probability = (double) distribution/instances.numInstances();
    		entropy += probability * log2(probability);
    	}
    	
    	// Return the entropy (in it's current sign).
    	return -entropy;
    }
    


	/**
     * Calculates the instances gini with respect to the their 
     * different class value.
     * @param  instances
     * @return the gini
     */
    public double calcGini(Instances instances) { 
		// Get class distribution
		Map<Double, Integer> classDistribution = this.classDistribution(instances);
		
		// Calculate the gini.
		double gini = 0;
		
		for (Map.Entry<Double, Integer> classEntry : classDistribution.entrySet()) {
			int distribution = classEntry.getValue();
			if (distribution == 0) {
				// Ignore non-existent class values.
				continue;
			}
			
			double probability = (double) distribution/instances.numInstances();
			
			gini += Math.pow(probability, 2);
		}
		
		// Return gini.
		return 1 - gini;
    }
    

    /**
     * Calculates the chi square statistic of splitting the data according 
     * to the splitting attribute as learned in class.
     * @param Instances object (a subset of the training data)
     * @param attributeIndex
     * @return Chi Square Statistic
     */
    public double calcChiSquare(Instances instances, int attributeIndex) {
    	// Split the instances according to the class values
    	Map<Double, Integer> classDistribution = this.classDistribution(instances);   //  <'+' : 7> , <'-' : 19> 
    	
    	// Split by the required attribute.
    	Map<Double, Instances> attributeSplit = this.splitByAttribute(instances, attributeIndex);  // <'hot': ins1, ins4, ins7> <'cold': ins2, ins3>
    	
    	double chiSquare = 0;
    	
    	// Iterate on every attribute value.
    	for (Map.Entry<Double, Instances> attributeSplitEntry : attributeSplit.entrySet()) {
    		Instances attributeValueInstances = attributeSplitEntry.getValue();
    		
    		// Calculte the formula for this attribute value, with every class value.
    		Map<Double, Integer> attributeValueClassDistribution = this.classDistribution(attributeValueInstances); //  hot ===> <'+' : 1> , <'-' : 2>
    		
    		for (Map.Entry<Double, Integer> attributeValueClassDistributionEntry : attributeValueClassDistribution.entrySet()) {
    			double classValue = attributeValueClassDistributionEntry.getKey();
    			int classValueCount = attributeValueClassDistributionEntry.getValue();
    			
    			
    			double expectation = attributeValueInstances.numInstances() * ((double) classDistribution.get(classValue) / instances.numInstances());
    			
    			if (expectation != 0 ) {
        			chiSquare += Math.pow(classValueCount - expectation, 2) / expectation;

    			}
    		}
    	}
    	
    	return chiSquare;
    }
    
    
	/**
	 * Print the decision tree.
	 */
	public void printTree() {
		System.out.println("Root");
		System.out.println("Returning value : " + this.rootNode.returnValue);		
		printTree(this.rootNode, "\t");
	}
	
	

	/*private methods */
	
    /**
     * Build the decision tree - Recursive implementation.
     * As long as the instances are still impure, keep splitting them according
     * to the best possible splitting attribute (that results in the 
     * highest gini/information gain). Wich should be null for the root node.  
     * @param parentNode - the parent of the current node. 
     * @param instances 
     * @param eImpurityMode - can be gini or impurity
     * @param branch - this argument keep the attribute value
     * @return desicion tree with respect to the impurityMeasure has given as an argument.
     */
	private Node buildTreeRec(Node parentNode, Instances instances, eImpurityMode impurityMode, double branch) {    
		Node currentNode = new Node();
	 
		currentNode.parent = parentNode;
		currentNode.branch = branch;                                                                               
		
		// Notice that even if the node is not a leaf, the return value
		// is the most prevalent class value among it's set of instances.
		currentNode.returnValue = this.getMostPrevalentClassValue(instances);
		
		// Calculte current impurity.
		double impurity = impurityMode.equals(eImpurityMode.Gini) ? 
				calcGini(instances) : calcEntropy(instances);
					
		// If the instances are perfectly classified already, then we are done
		// splitting this subset.
		if (impurity == 0) {
			return currentNode;
		}
		
		// Else, find the best attribute for splitting all the instances.
		int maxGainAttributeIndex = -1;
		double maxGain = Double.NEGATIVE_INFINITY;
		
		for (int attributeIndex = 0; attributeIndex < instances.numAttributes(); attributeIndex++) {
			if (instances.classIndex() == attributeIndex) {
				// Do not try to split according to the attribute class value.  
				continue;
			}
		
			double currentGain = impurityMode.equals(eImpurityMode.Gini) ? 
					this.calcGiniGain(instances, attributeIndex) : this.calcInfoGain(instances, attributeIndex);
						
			if (currentGain > maxGain) {
				// Update the best attribute and his gain.
				maxGainAttributeIndex = attributeIndex;
				maxGain = currentGain;
			}			
		}
		
		// Update best attribute index of this current node
	    currentNode.attributeIndex = maxGainAttributeIndex; 
		
		// Verify that we have indeed gained new information by splitting 
		// the instances (according to any attribute). 
		if (maxGain <= 0) {
			// If we couldn't gain any information by splitting the instances
			// according to any attribute, it means that we have "noise" 
			// in the data, and further splitting is not possiblle.
			// Thus, this node should be a leaf anyway.
			return currentNode;
		}
		
		// Check if required to prune by Chi Square.
		if (this.m_PruningMode.equals(ePruningMode.Chi)) {
			
			int degOfFreedom = this.degOfFreedom(instances, maxGainAttributeIndex);      	
			int PValueIndex = this.getPValueIndex();
			double[][] tableOfChiProb =  this.getTableOfChiProb();
			double chiSquaredReferenceValue = tableOfChiProb[degOfFreedom - 1][PValueIndex];
						
			// Compare actual statistic calculation with the probability.
			if (this.calcChiSquare(instances, maxGainAttributeIndex) < chiSquaredReferenceValue) { 
				
				// Chi Square is statistic is to low. Prune by avoiding 
				// further splituing the node.
				return currentNode;
			}
		}
		
		// Split the instances according to the found attribute. 
		Map<Double, Instances> attributeSplit = this.splitByAttribute(instances, maxGainAttributeIndex);
				
		// Recursively operate on the instances subsets and create 
		// children nodes from them.
		List<Node> childrenNodes = new ArrayList<Node>();
		currentNode.attributeValuesBranches = new ArrayList<Double>();     
		
		for (Map.Entry<Double, Instances> attributeValueEntry : attributeSplit.entrySet()) {
			
			// Add the attribute value to the branches list of the node.
			double attributeValue = attributeValueEntry.getKey();
			currentNode.attributeValuesBranches.add(attributeValue);
																					     	                         	              
			Instances attributeValueSubSet = attributeValueEntry.getValue();
		
			// Create child node and add it to the list of children.
			Node childNode = this.buildTreeRec(currentNode, attributeValueSubSet, impurityMode, attributeValue);
		
			childrenNodes.add(childNode);
		
		}
		
		// Set children nodes of current node
		currentNode.children = childrenNodes.toArray(new Node[childrenNodes.size()]);
		
		return currentNode;
	}

	
	
	/**
	 * Calc the value of degree of freedom.
	 * @param instances
	 * @param attributeIndex
	 * @return degree of freedom value
	 */
	private int degOfFreedom(Instances instances, int attributeIndex) {
		Map<Double, Instances> attributeSplit = this.splitByAttribute(instances, attributeIndex);  // <small : inst1, inst3 > , <medium : inst2, inst4>, <large : >
		
		int counter = 0;
		
		for (Map.Entry<Double, Instances> attributeValueEntry : attributeSplit.entrySet()) {
			
			int numOfIstrances = attributeValueEntry.getValue().numInstances();
			
			if (numOfIstrances != 0) {
				counter ++;
			}
		}
		
		int degOfFreedom = counter - 1;
		
		return degOfFreedom;
	}
	
	
	
	/**
	 * Split the instances according to the different attribute values.
	 * Each attribute value is mapped to the subset of the instances 
	 * that have that specific value.
	 * @param instances
	 * @param attributeIndex
	 * @return HashMap that maps each attribute value to an instances object containing all the instances of that attribute value.
	 */
	private Map<Double, Instances> splitByAttribute(Instances instances, int attributeIndex) {
		// Map of attribute values to instances with these attribute values.
		Map<Double, Instances> attributeSplit = new HashMap<Double, Instances>();
		
		// Iterate through the instances and map the attribute values to them.
		for (int i = 0 ; i < instances.numInstances(); i++) {
			Instance instance = instances.instance(i);
			double attributeValue = instance.value(attributeIndex);
			
			Instances attributeValueSubset = attributeSplit.get(attributeValue);
			if (attributeValueSubset == null) {
				// We haven't seen this attribute value before.
				// Create an empty attribute value subset, and make 
				// the current instance the only element in it.
				attributeValueSubset = new Instances(instances, 0);
				attributeValueSubset.add(instance);
				attributeSplit.put(attributeValue, attributeValueSubset);
			} else {
				// Add current instance to that attribute value subset.
				attributeValueSubset.add(instance);
			}
		}	
		
		return attributeSplit;
	}
	
	
	
	/**
	 * Calculate the information gain, if splitting the instances according to 
	 * the given attribute. This is done by reducing the weighted entropy of 
	 * the resulted subsets from the parent entropy.
	 * @param instances
	 * @param attributeIndex
	 * @return information gain value
	 */
	private double calcInfoGain(Instances instances, int attributeIndex) {
		double parentEntropy = calcEntropy(instances);
		
		// Split parent instances according to given attribute to 
		// different instances subsets.
		Map <Double, Instances> attributeSplit = this.splitByAttribute(instances, attributeIndex);
		
		// Calculate the weighted entropies of the subsets.
		double weightedSubsetsEntropies = 0;
		
		for (Map.Entry<Double, Instances> attributeValueEntry : attributeSplit.entrySet()) {
			Instances attributeValueSubset = attributeValueEntry.getValue();
			
			// Calculate subset weight.
			double subsetWeight = (double) attributeValueSubset.numInstances() / instances.numInstances();
			double subsetEntropy = this.calcEntropy(attributeValueSubset);
			
			weightedSubsetsEntropies += subsetWeight * subsetEntropy;
		}
		
		// Calculate and return the information gain.
		return parentEntropy - weightedSubsetsEntropies;
	}
	
	
	
	/**
	 * Calculate the gini gain, if splitting the instances according to 
	 * the given attribute. This is done by reducing the weighted gini of 
	 * the resulted subsets from the parent entropy.
	 * @param instances
	 * @param attributeIndex
	 * @return gini gain value
	 */
	private double calcGiniGain(Instances instances, int attributeIndex) {
		double parentGini = calcGini(instances);
		
		// Split parent instances according to given attribute to 
		// different instances subsets.
		Map <Double, Instances> attributeSplit = this.splitByAttribute(instances, attributeIndex);
		
		// Calculate the weighted ginies of the subsets.
		double weightedSubsetsGinies = 0;
		
		for (Map.Entry<Double, Instances> attributeValueEntry : attributeSplit.entrySet()) {
			Instances attributeValueSubset = attributeValueEntry.getValue();
			
			// Calculate subset weight.
			double subsetWeight = (double) attributeValueSubset.numInstances() / instances.numInstances();
			double subsetGini = this.calcGini(attributeValueSubset);
			
			weightedSubsetsGinies += subsetWeight * subsetGini;
		}
		
		// Calculate and return the information gain.
		return parentGini - weightedSubsetsGinies;
	}
	
	
	/**
	 * Find the most prevalent class value among the given instances.
	 * @param instances
	 * @return Most prevalent class value
	 */
	private double getMostPrevalentClassValue(Instances instances) {
		Map<Double, Integer> classDistribution = this.classDistribution(instances);
		double mostPrevalentClassValue = -1;
		int mostPrevalentClassValueCount = 0;
		
		// Find the most prevalent class value among the given instances.
		for (Map.Entry<Double, Integer> classDistributionEntry : classDistribution.entrySet()) {
			double classValue = classDistributionEntry.getKey();
			int classValueCount = classDistributionEntry.getValue();
		
			if (classValueCount > mostPrevalentClassValueCount) {
				// Found more prevalent class value, update our findings.
				mostPrevalentClassValue = classValue;
				mostPrevalentClassValueCount = classValueCount;
			}
		}
		
		return mostPrevalentClassValue;
	}
	
	
	/**
	 * Compute the classes distribution among the given instances.
	 * Each possible class value is mapped to its count among the given instances.
	 * @param instances
	 * @return HashMap that maps each class value to its count in the instances.
	 */
    private Map<Double, Integer> classDistribution(Instances instances) {
	
		return attributeDistribution(instances, instances.classIndex());
	}

	
	/**
	 * Compute the attribute distribution among the given instances.
	 * Each possible attribute value is mapped to its count among the given instances.
	 * This can also be used to count the class distribution, if given the class index.
	 * @param instances
	 * @param attributeIndex
	 * @return Map that maps each class value to its count in the instances.
	 */
	private Map<Double, Integer> attributeDistribution(Instances instances, int attributeIndex) {
		
		Map<Double, Integer> attributeDistribution = new HashMap<Double, Integer>();
		
		// Start by setting the mapping of all possible attribute values to 0
		// NOTE: This works only for nominal attributes!!!
		// (In case of non-nominal attributes, we will not have any zero-counted attribute value).
		Attribute attribute = instances.attribute(attributeIndex);
		
		for (int i = 0; i < attribute.numValues(); i++) {
			attributeDistribution.put((double)i, 0);
		}
		
		// Count appearances of the different attribute values.
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance instance = instances.instance(i);
			double attributeValue = instance.value(attributeIndex);
			
			Integer count = attributeDistribution.get(attributeValue);
			
			if (count == null) {
				// We haven't seen this attribute value before, set it's count to 1.
				attributeDistribution.put(attributeValue, 1);
			} else {
				// Update this attribute value count.
				attributeDistribution.put(attributeValue, count + 1);
			}
		}
		
		return attributeDistribution;
	}
	
	
	
	/**
	 * Ptint the decision tree recursivly.
	 * @param node
	 * @param t  tab
	 */
	private void printTree(Node node, String t) {
				
		if (node.children == null) {
			System.out.println(t + "\t");
			System.out.println("Leaf. Returning value: " + node.returnValue);
			return;
		}
			
		for (int i = 0; i < node.children.length; i++ ) {
		
			System.out.println(t + "If attribute" + node.attributeIndex + " = " + node.children[i].branch);
		
			
			printTree(node.children[i], t + "\t");
		}
	}
		
	
	/* Default and helper functions */
	
    /**
     * calc log with base 2
     * @param n
     * @return
     */
    private double log2(double n) {
		
		return (Math.log(n) / Math.log(2));
	}

	
	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}