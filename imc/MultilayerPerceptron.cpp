

#include "MultilayerPerceptron.h"

#include "util.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>
#include <cstring>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Obtain an integer random number in the range [Low,High]
int randomInt(int Low, int High)
{
	
}

// ------------------------------
// Obtain a real random number in the range [Low,High]
double randomDouble(double Low, double High)
{

}

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	this->eta=0.7;
	this->mu=1;
	this->validationRatio=0.0;
	this->decrementFactor=1;
	this->online=false;
	this->outputFunction=0;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {

	this->nOfLayers = nl;

	this->layers = new Layer[nOfLayers];
	if (this->layers == nullptr) {
		std::cerr << "Error al reservar memoria" << std::endl;
		return -1;
	}

	for (int i = 0; i < nOfLayers; i++) {
		layers[i].nOfNeurons = npl[i];
		layers[i].neurons = new Neuron[npl[i]];
		if (layers[i].neurons == nullptr) {
			std::cerr << "Error al reservar memoria " << std::endl;
			return -1;
		}

		for (int j = 0; j < npl[i]; j++) {
			if (i == 0) {
				layers[i].neurons[j].w = nullptr;
				layers[i].neurons[j].deltaW = nullptr;
				layers[i].neurons[j].lastDeltaW = nullptr;
				layers[i].neurons[j].wCopy = nullptr;
			}
			else {
				layers[i].neurons[j].w = new double[npl[i-1] + 1];
				layers[i].neurons[j].deltaW = new double[npl[i-1] + 1];
				layers[i].neurons[j].lastDeltaW = new double[npl[i-1] + 1];
				layers[i].neurons[j].wCopy = new double[npl[i-1] + 1];

				if (layers[i].neurons[j].w          == nullptr ||
					layers[i].neurons[j].deltaW     == nullptr ||
					layers[i].neurons[j].lastDeltaW == nullptr ||
					layers[i].neurons[j].wCopy      == nullptr
				){
					std::cerr << "Error al reservar memoria" << std::endl;
					return -1;
				}
			}	
		}
	}


	return 1;

}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {
	for (int i = 0; i < nOfLayers; i++) {
		for (int j = 0; j < layers[i].nOfNeurons; j++) {
			delete[] layers[i].neurons[j].w;
			delete[] layers[i].neurons[j].deltaW;
			delete[] layers[i].neurons[j].lastDeltaW;
			delete[] layers[i].neurons[j].wCopy;
		}
		delete[] layers[i].neurons;
	}
	
	delete[] layers;
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {

	for (int i = 1; i < nOfLayers; i++) {
		for (int j = 0; j < layers[i].nOfNeurons; j++) {
			layers[i].neurons[j].w[0]=0;
			for (int k = 0; k < layers[i-1].nOfNeurons + 1; k++) {
				layers[i].neurons[j].w[k] = ((rand() % 200000) / 100000.0)-1;
				layers[i].neurons[j].deltaW[k] = 0.0;
				layers[i].neurons[j].lastDeltaW[k] = 0.0;
				layers[i].neurons[j].wCopy[k] = 0.0;
			}
		}
	}

}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) {

	for (int i = 0; i < layers[0].nOfNeurons; i++) {
		layers[0].neurons[i].out = input[i];
	}

}

// ------------------------------
// Get the outputs predicted by the network (out vector of the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output)
{

	for (int i = 0; i < layers[nOfLayers-1].nOfNeurons; i++) {
		output[i] = layers[nOfLayers-1].neurons[i].out;
	}

}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {

	for (int i = 1; i < nOfLayers; i++) {
		for (int j = 0; j < layers[i].nOfNeurons; j++) {
			for (int k=0; k<layers[i-1].nOfNeurons;k++){
				layers[i].neurons[j].wCopy[k]=layers[i].neurons[j].w[k];
			};
		}
	}	

}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {

	for (int i = 1; i < nOfLayers; i++) {
		for (int j = 0; j < layers[i].nOfNeurons; j++) {
			for (int k=0; k<layers[i-1].nOfNeurons;k++){
				layers[i].neurons[j].w[k]=layers[i].neurons[j].wCopy[k];
			};
		}
	}

}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {

	for(int i=1; i<nOfLayers; i++){
		if(i==nOfLayers-1 && outputFunction == 1){
			double sumaNet = 0.0;
			for (int j=0; j<layers[i].nOfNeurons;j++){
				double suma = 0.0;
				for (int k=1; k<layers [i-1].nOfNeurons + 1; k++){
					suma += layers[i].neurons[j].w[k] * layers[i-1].neurons[k-1].out;
				}
				sumaNet += exp(suma);
			}
			for (int j=0; j<layers[i].nOfNeurons;j++){
				double suma = 0.0;
				for (int k=1; k<layers [i-1].nOfNeurons + 1; k++){
					suma += layers[i].neurons[j].w[k] * layers[i-1].neurons[k-1].out;
				}
				layers[i].neurons[j].out = exp(suma) / sumaNet;
			}
		}
		else{
			for (int j=0; j<layers[i].nOfNeurons;j++){
				double suma = layers[i].neurons[j].w[0];
				for (int k = 0; k < layers[i-1].nOfNeurons; k++) {
					suma = suma + layers[i-1].neurons[k].out * layers[i].neurons[j].w[k+1];
				}
				layers[i].neurons[j].out = 1.0 / (1.0 + exp(-suma));
			} 
		}
	}

}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(double* target, int errorFunction) {

	if(errorFunction == 0){
		double MSE = 0.0;
		for (int i = 0; i < layers[nOfLayers-1].nOfNeurons; i++) {
			double diferencia = (target[i] - layers[nOfLayers-1].neurons[i].out);
			MSE = MSE+( diferencia * diferencia);
		}
		MSE /= layers[nOfLayers-1].nOfNeurons;
		return MSE;	
	}
	else{
		double entropy=0.0;
		for (int i = 0; i < layers[nOfLayers-1].nOfNeurons; i++) {

			if(target[i]==1){
				entropy+=log(layers[nOfLayers-1].neurons[i].out);
			}



			/*double opo=layers[nOfLayers-1].neurons[i].out;
			cout<<"Opo: "<<opo<<endl;
			cout<<"entropy=entropy + "<<target[i]<<"*"<<log(opo)<<endl;
			entropy = entropy + (target[i]*log(opo));*/
		}
		entropy /= (layers[nOfLayers-1].nOfNeurons);
		return entropy;
	}


}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(double* target, int errorFunction) {

	for (int i = nOfLayers - 1; i >= 0; i--)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			double o = layers[i].neurons[j].out;
			if (i == nOfLayers - 1)
			{
				if (outputFunction == 0 && errorFunction == 0){
					layers[i].neurons[j].delta = (target[j] - o) * o * (1 - o);
				}
				else if (outputFunction == 0 && errorFunction == 1){
					double divisor;
						if (std::fabs(0-o)<0.000000001){
							//cout<<"ENTRO AQUÍ"<<endl;
							divisor = 0.000000001;
						}
						else{
							divisor=o;
						}
					layers[i].neurons[j].delta = (target[j] / divisor) * o * (1 - o);
				}
				else if (outputFunction == 1 && errorFunction == 0)
				{
					double suma = 0.0;
					for (int k = 0; k < layers[i].nOfNeurons; k++)
					{
						if (k == j)
							suma += (target[k] - layers[i].neurons[k].out) * o * (1 - layers[i].neurons[k].out);
						else
							suma += (target[k] - layers[i].neurons[k].out) * o * (-layers[i].neurons[k].out);
					}
					layers[i].neurons[j].delta = suma;
				}
				else
				{
					double suma = 0.0;
					for (int k = 0; k < layers[i].nOfNeurons; k++)
					{
						double divisor;
						if (std::fabs(0-layers[i].neurons[k].out)<0.000000001){
							cout<<"ENTRO AQUÍ"<<endl;
							divisor = 0.000000001;
						}
						else{
							divisor=layers[i].neurons[k].out;
						}
						/*cout<<"target[k]="<<target[k]<<endl;
						cout<<"divisor="<<divisor<<endl;
						cout<<"o="<<o<<endl;
						cout<<"layers[i].neurons[k].out="<<layers[i].neurons[k].out<<endl;*/
						if (k == j)
							suma += (target[k] / divisor) * o * (1 - layers[i].neurons[k].out);
						else
							suma += (target[k] / divisor) * o * (-layers[i].neurons[k].out);
					}
					//cout<<"Sigmas de salida: "<<suma<<endl;
					//cout<<"Salida: "<<layers[i].neurons[j].out<<endl;
					layers[i].neurons[j].delta = suma;
				}
			}
			else
			{
				double suma = 0.0;
				for (int k = 0; k < layers[i + 1].nOfNeurons; k++)
				{
					suma += layers[i + 1].neurons[k].delta * layers[i + 1].neurons[k].w[j + 1];
				}
				layers[i].neurons[j].delta = suma * o * (1 - o);
				//cout<<"Sigmas no de salida: "<<layers[i].neurons[j].delta<<endl;
			}
		}
	}
	

}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {
	for (int i = 1; i < nOfLayers; i++) {
		for (int j = 0; j < layers[i].nOfNeurons; j++) {
			for (int k = 0; k < layers[i-1].nOfNeurons; k++) {
				layers[i].neurons[j].deltaW[k+1] += layers[i].neurons[j].delta * layers[i-1].neurons[k].out;
			}
			layers[i].neurons[j].deltaW[0] += layers[i].neurons[j].delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {
	for (int i = 1; i < nOfLayers; i++) {
		double eta_i = eta * pow(decrementFactor, -(nOfLayers-1-i));
		for (int j = 0; j < layers[i].nOfNeurons; j++) {
			for (int k = 0; k < layers[i-1].nOfNeurons+1; k++) {
				layers[i].neurons[j].w[k] += eta_i * layers[i].neurons[j].deltaW[k] + eta_i * mu * layers[i].neurons[j].lastDeltaW[k];
				layers[i].neurons[j].lastDeltaW[k] = layers[i].neurons[j].deltaW[k];
				layers[i].neurons[j].deltaW[k] = 0.0;

			}
		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
	for (int i = 1; i < nOfLayers; i++) {
		std::cout << "CAPA " << i<< std::endl;
		for (int j = 0; j < layers[i].nOfNeurons; j++) {
			std::cout << "[ ";
			for (int k = 0; k < layers[i-1].nOfNeurons+1; k++) {
				std::cout << layers[i].neurons[j].w[k] << " ";
			}
			std::cout << " ]" << std::endl;
		}
	}

}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(double* input, double* target, int errorFunction) {

	feedInputs(input);
	forwardPropagate();
	backpropagateError(target, errorFunction);
	accumulateChange();
	if (online){
		weightAdjustment();
	}
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset* MultilayerPerceptron::readData(const char *fileName) {

	ifstream myFile(fileName);    // Create an input stream

    if (!myFile.is_open()) {
       cout << "ERROR: I cannot open the file " << fileName << endl;
       return nullptr;
    }

	Dataset* dataset = new Dataset;
	if(dataset == nullptr)
		return nullptr;

	string line;

	if(myFile.good()) {
		getline(myFile, line);   // Read a line
		istringstream iss(line);

		iss >> dataset->nOfInputs;
		iss >> dataset->nOfOutputs;
		iss >> dataset->nOfPatterns;
	}
	dataset->inputs = new double*[dataset->nOfPatterns];
	dataset->outputs = new double*[dataset->nOfPatterns];

	for(int i = 0; i < dataset->nOfPatterns; i++){
		dataset->inputs[i] = new double[dataset->nOfInputs];
		dataset->outputs[i] = new double[dataset->nOfOutputs];
	}

	int i = 0;
	while (myFile.good()) {
		getline(myFile, line);   // Read a line
		if (!line.empty()) {
			istringstream iss(line);
			for(int j = 0; j < dataset->nOfInputs; j++){
				double value;
				iss >> value;
				if(!iss)
					return nullptr;
				dataset->inputs[i][j] = value;
			}
			for(int j = 0; j < dataset->nOfOutputs; j++){
				double value;
				iss >> value;
				if(!iss)
					return nullptr;
				dataset->outputs[i][j] = value;
			}
			i++;
		}
	}

	myFile.close();

	return dataset;

}


// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction) {
	int i;
	for(i=0; i<trainDataset->nOfPatterns; i++){
		performEpoch(trainDataset->inputs[i], trainDataset->outputs[i],errorFunction);
	}
	if(!online){
		for (int i = 1; i<nOfLayers;i++){
			for(int j=0; j<layers[i].nOfNeurons;j++){
				for(int k=0;k<layers[i-1].nOfNeurons+1;k++){
					layers[i].neurons[j].deltaW[k] /= trainDataset->nOfPatterns;
				}
			}
		}
		weightAdjustment();
	}
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset* dataset, int errorFunction) {

	double error = 0.0;
	for (int i = 0; i < dataset->nOfPatterns; i++) {
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		error += obtainError(dataset->outputs[i], errorFunction);
	}

	error /= dataset->nOfPatterns;
	if (errorFunction==1)
		error=-error;
	
	return error;

}


// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset* dataset) {

	double ccr=0.0;

	int **matrix= new int *[dataset->nOfPatterns];
	for(int i = 0; i< dataset->nOfOutputs;i++){
		matrix[i] = new int[dataset->nOfPatterns];
		for(int j = 0; j< dataset-> nOfOutputs;j++){
			matrix[i][j]=0;
		}
	}

	for(int i=0; i<dataset->nOfPatterns;i++){
		feedInputs(dataset->inputs[i]);
		forwardPropagate();

		int target=0;

		int prediccion = 0;

		for (int j=1; j<dataset->nOfOutputs; j++){
			if(layers[nOfLayers-1].neurons[j].out > layers[nOfLayers-1].neurons[target].out){
				target=j;
			}
			if(dataset->outputs[i][j] > dataset->outputs[i][prediccion]){
				prediccion = j;
			}
			matrix[target][prediccion]=matrix[target][prediccion]+1;
		}

		ccr += (target == prediccion);
	}

	ccr *= 100.0 / dataset->nOfPatterns;

	for(int i= 0; i<dataset->nOfOutputs;i++){
		cout<<"(";
		for (int j = 0; j< dataset -> nOfOutputs; j++){
			cout << " "<<matrix[i][j];
		}
		cout<<")"<<endl;
	}

	return ccr;

}


// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset
void MultilayerPerceptron::predict(Dataset* dataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * salidas = new double[numSalidas];
	
	cout << "Id,Category" << endl;
	
	for (i=0; i<dataset->nOfPatterns; i++){

		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;
		
		cout << i << "," << maxIndex << endl;

	}
}



// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// Both training and test CCRs should be obtained and stored in ccrTrain and ccrTest
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::runBackPropagation(Dataset * trainDataset, Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int errorFunction)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving = 0;
	nOfTrainingPatterns = trainDataset->nOfPatterns;

	Dataset * validationDataset = NULL;
	double validationError = 0, previousValidationError = 0;
	int iterWithoutImprovingValidation = 0;

	// Generate validation data
	if(validationRatio > 0 && validationRatio < 1){
		validationDataset = new Dataset ();
		validationDataset->nOfPatterns = (int)(validationRatio * trainDataset->nOfPatterns);
			if (validationDataset->nOfPatterns > 0) {
				
				trainDataset->nOfPatterns -= validationDataset->nOfPatterns;

				validationDataset->nOfInputs = trainDataset->nOfInputs;
				validationDataset->nOfOutputs = trainDataset->nOfOutputs;

				validationDataset->inputs = new double*[validationDataset->nOfPatterns];
				validationDataset->outputs = new double*[validationDataset->nOfPatterns];

				if (validationDataset->inputs == nullptr || validationDataset->outputs == nullptr) {
					std::cerr << "No se pudo reservar memoria para el dataset de validación" << std::endl;

					//Cancel validation since we couldn't reserve memory
					validationRatio = 0;
					trainDataset->nOfPatterns += validationDataset->nOfPatterns;
				}
				else {
					//Copy the last n patterns (pointers) from trainDataset to validationDataset
					std::memcpy(validationDataset->inputs,
								trainDataset->inputs+trainDataset->nOfPatterns,
								sizeof(double*) * validationDataset->nOfPatterns
					);
					std::memcpy(validationDataset->outputs,
								trainDataset->outputs+trainDataset->nOfPatterns,
								sizeof(double*) * validationDataset->nOfPatterns
					);
				}
			
			}
			else {
				validationRatio = 0;
			}
	}

	// Learning
	do {

		train(trainDataset,errorFunction);

		double trainError = test(trainDataset,errorFunction);
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if(iterWithoutImproving==50){
			cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}

		countTrain++;

		if(validationDataset!=NULL){
			if(previousValidationError==0)
				previousValidationError = 999999999.9999999999;
			else
				previousValidationError = validationError;
			validationError = test(validationDataset,errorFunction);
			if(validationError < previousValidationError)
				iterWithoutImprovingValidation = 0;
			else if((validationError-previousValidationError) < 0.00001)
				iterWithoutImprovingValidation = 0;
			else
				iterWithoutImprovingValidation++;
			if(iterWithoutImprovingValidation==50){
				cout << "We exit because validation is not improving!!"<< endl;
				restoreWeights();
				countTrain = maxiter;
			}
		}

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << validationError << endl;

	} while ( countTrain<maxiter );

	if ( (iterWithoutImprovingValidation!=50) && (iterWithoutImproving!=50))
		restoreWeights();

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		double* prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	*errorTest=test(testDataset,errorFunction);;
	*errorTrain=minTrainError;
	*ccrTest = testClassification(testDataset);
	*ccrTrain = testClassification(trainDataset);

}

// -------------------------
// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * fileName)
{
	// Object for writing the file
	ofstream f(fileName);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
	{
		f << " " << layers[i].nOfNeurons;
	}
	f << " " << outputFunction;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(layers[i].neurons[j].w!=NULL)
				    f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// -----------------------
// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * fileName)
{
	// Object for reading a file
	ifstream f(fileName);

	if(!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
	{
		f >> npl[i];
	}
	f >> outputFunction;

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(!(outputFunction==1 && (i==(nOfLayers-1)) && (k==(layers[i].nOfNeurons-1))))
					f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
