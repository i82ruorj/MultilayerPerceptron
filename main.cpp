#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <float.h> // For DBL_MAX

#include "imc/MultilayerPerceptron.h"


using namespace imc;
using namespace std;

int main(int argc, char **argv) {
	// Process the command line
    bool Tflag = 0, wflag = 0, pflag = 0, iflag=0, lflag=0, hflag=0, eflag=0, mflag=0, dflag=0, vflag=0, tflag=0, oflag=0, fflag=0, sflag=0;
    char *Tvalue = NULL, *wvalue = NULL, *tvalue = NULL, *ivalue = NULL, *lvalue = NULL, *hvalue = NULL, *evalue = NULL, *mvalue = NULL, *vvalue = NULL, *dvalue = NULL, *pvalue = NULL, *fvalue=NULL;
    int c;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:w:i:l:h:e:m:v:d:f:ops")) != -1)
    {

        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
            case 't':
                tflag = true;
                tvalue = optarg;
                break;
            case 'i':
                iflag = true;
                ivalue = optarg;
                break;
            case 'l':
                lflag = true;
                lvalue = optarg;
                break;
            case 'h':
                hflag = true;
                hvalue = optarg;
                break;
            case 'e':
                eflag = true;
                evalue = optarg;
                break;
            case 'm':
                mflag = true;
                mvalue = optarg;
                break;
            case 'v':
                vflag = true;
                vvalue = optarg;
                break;
            case 'd':
                dflag = true;
                dvalue = optarg;
                break;
            case 'p':
                pflag = true;
                pvalue = optarg;
                break;
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'o':
                oflag = true;
                break;
            case 'f':
                fflag = true;
                fvalue = optarg;
                break;
            case 's':
                sflag = true;
                break;
            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 'p'|| optopt =='t' || optopt =='i' || optopt == 'l'|| optopt == 'h' || optopt == 'e' || optopt == 'm' || optopt == 'v' || optopt == 'd' || optopt =='f')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown character `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }


    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value

    	// Type of error considered
        

        int error=0;

        if(fflag){
            error=atoi(fvalue);
        }

        int function=0;
        if(sflag){
            function=1;
        }

    	// Maximum number of iterations
    	int maxIter=500;
        if (iflag){
            maxIter=atoi(ivalue);
        } // This should be completed
        std::cout<<"hola2"<<std::endl;

        // Read training and test data: call to mlp.readData(...)
    	Dataset * trainDataset = mlp.readData(tvalue); // This should be corrected
        Dataset * testDataset = mlp.readData(tvalue); // This should be corrected
        if (Tflag){
    	    testDataset = mlp.readData(Tvalue); // This should be corrected
        }

        // Initialize topology vector
        std::cout<<"hola3"<<std::endl;

        int layers;
        if(lflag){
            layers=atoi(lvalue); // This should be corrected
        }
        else{
            layers=1; 
        }
        double eta, mu,v,d;
        bool online = false;
        std::cout<<"hola4"<<std::endl;

        if (oflag){
            online=true;
        }
        if (eflag){
            eta=atof(evalue);
        }
        else{
            eta=0.7;
        }
    	if (mflag){
            mu=atof(mvalue);
        }
        else{
            mu=1;
        }
        if (vflag){
            v=atof(vvalue);
        }
        else{
            v=0.0;
        }if (dflag){
            d=atof(dvalue);
        }
        else{
            d=1.0;
        }
        int neurons = 5;
        if(hflag){
            neurons=atoi(hvalue);
        }

        int *topology = new int[layers+2];
        topology[0] = trainDataset->nOfInputs;
        for(int i=1; i<(layers+1); i++)
            topology[i] = neurons;
        topology[layers+1] = trainDataset->nOfOutputs;
        mlp.initialize(layers+2,topology);
        mlp.eta=eta;
        mlp.mu=mu;
        mlp.validationRatio = v;
        mlp.decrementFactor = d;
        mlp.online=online;
        mlp.outputFunction=function;
        std::cout<<"hola"<<std::endl;

		// Seed for random numbers
		int seeds[] = {1,2,3,4,5};
		double *trainErrors = new double[5];
		double *testErrors = new double[5];
		double *trainCCRs = new double[5];
		double *testCCRs = new double[5];
		double bestTestError = DBL_MAX;
        double trainAverageError = 0, trainStdError = 0;
		double testAverageError = 0, testStdError = 0;
		double trainAverageCCR = 0, trainStdCCR = 0;
		double testAverageCCR = 0, testStdCCR = 0;

		for(int i=0; i<5; i++){
			cout << "**********" << endl;
			cout << "SEED " << seeds[i] << endl;
			cout << "**********" << endl;
			srand(seeds[i]);
			mlp.runBackPropagation(trainDataset,testDataset,maxIter,&(trainErrors[i]),&(testErrors[i]),&(trainCCRs[i]),&(testCCRs[i]),error);
			cout << "We end!! => Final test CCR: " << testCCRs[i] << endl;

			// We save the weights every time we find a better model
			if(wflag && testErrors[i] <= bestTestError)
			{
				mlp.saveWeights(wvalue);
				bestTestError = testErrors[i];
			}

            testAverageError += testErrors[i];
            trainAverageError += trainErrors[i];
            testAverageCCR += testCCRs[i];
            trainAverageCCR += trainCCRs[i];
		}


		testAverageError /= 5;
        trainAverageError /= 5;
        testAverageCCR /= 5;
        trainAverageCCR /= 5;


        for (int i = 0; i < 5; i++)
        {
            trainStdError += pow((trainErrors[i] - trainAverageError), 2);
            testStdError += pow((testErrors[i] - testAverageError), 2);
            trainStdCCR += pow((trainCCRs[i] - trainAverageCCR), 2);
            testStdCCR += pow((testCCRs[i] - testAverageCCR), 2);
        }
        trainStdError /= 5;
        testStdError /= 5;
        trainStdCCR /= 5;
        testStdCCR /= 5;
        trainStdError = sqrt(trainStdError);
        testStdError = sqrt(testStdError);
        trainStdCCR = sqrt(trainStdCCR);
        testStdCCR = sqrt(testStdCCR);

        // Obtain training and test averages and standard deviations

		cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

		cout << "FINAL REPORT" << endl;
		cout << "*************" << endl;
	    cout << "Train error (Mean +- SD): " << trainAverageError << " +- " << trainStdError << endl;
	    cout << "Test error (Mean +- SD): " << testAverageError << " +- " << testStdError << endl;
	    cout << "Train CCR (Mean +- SD): " << trainAverageCCR << " +- " << trainStdCCR << endl;
	    cout << "Test CCR (Mean +- SD): " << testAverageCCR << " +- " << testStdCCR << endl;






		return EXIT_SUCCESS;
    } else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // You do not have to modify anything from here.
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to mlp.readData(...)
        Dataset *testDataset;
        testDataset = mlp.readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;

	}
}

