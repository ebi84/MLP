#include "Header.h"



int main()
{
	double begin = clock();

	//srand((int)time(NULL)); 
	srand(0);
	int STATUS;

	int num_train(1797), num_feature(64);
	int	num_output(10), num_iteration(600);  
	//omp_set_num_threads(12);

	float* X_train = new float[num_feature * num_train];              // allocating memory for X_train
	int* Y_train = new int[num_train];                                // allocating memory for Y_train

	LoadData(X_train, Y_train, num_feature, num_train);               // loading data from csv file
	Normalize(X_train, num_feature, num_train);                       // normalizing data

	int num_hidden1(10), num_hidden2(5);

	DeepModel myModel(num_feature, num_train, num_output);            // creating a DeepModel
	                                                                  
																     
	STATUS = myModel.AddLayer("Dense", num_hidden1, "relu");          // adding a dense layer with relu activation
	if (STATUS) { return STATUS; }
	STATUS = myModel.AddLayer("Dense", num_hidden2, "relu");          // adding another dense layer with relu activation
	if (STATUS) { return STATUS; }
	STATUS = myModel.AddLayer("Dense", num_output, "softmax");        // adding the final layer with softmax activation
	if (STATUS) { return STATUS; }

	myModel.SetOptimizer("GradientDescent", 1.0f, 0.001f);            // set Gradient Descent for training
	myModel.Train(X_train, Y_train, num_iteration, "cross_entropy");  // Train. Comment this if don't need training
	float accuracy = myModel.Validate(X_train, Y_train, num_train);   // num_test cannot be bigger than num_train!
	display(accuracy);

	int num_test(10);  
	int* Y_pred = new int[num_test];                                  // allocate vector for classification result
	myModel.Classify(X_train, Y_pred, num_test);                      // classifies the first "num_test" samples of
	cout << "Predicted output: ";  dispVec(Y_pred, num_test);         // X_train. results are stored in Y_pred
	cout << "Real output:      ";  dispVec(Y_train, num_test);

	double end = clock();
	double calculation_time = (end - begin) / CLOCKS_PER_SEC;
	display(calculation_time);

	delete[] X_train;
	delete[] Y_train;
	delete[] Y_pred;

	return 0;
}