#pragma once

#ifndef HEADER_H
#define HEADER_H

#include<iostream>
#include<stdio.h>
#include<vector>
#include<string>
#include <iomanip>
#include<math.h>
#include<fstream>
#include<ctime>
#include<omp.h>

using namespace std;

#define MAX(a,b) a>b? a : b
#define MIN(a,b) a<b? a : b
#define display(name) printer(#name, (name))

template<typename T> //to print the value and name of a variable
void printer(const char* name, T value) {
	cout << name << " = " << value << endl;
}


template<typename T>
void dispVec(T* B, int size) //to display a vector
{
	for (int i = 0; i < size; i++)
	{
		cout << setw(3) << B[i] << " ";
	}
	cout << endl;
}


auto LoadData(float* X, int* Y, int num_feature, int num_train)->void;
auto Normalize(float* X, int num_feature, int num_train)->void;


class DeepModel
{

public:

	DeepModel(int t_num_feature, int t_num_train, int t_num_output); // constructor
	~DeepModel(); // destructor
	DeepModel(const DeepModel& other) = delete;

	vector<float>costs; // stores costs during training

	auto SetCostFunction(string type)->void;
	auto CalcCostFunction(int* Y)->void;
	auto SetOptimizer(string optimizer, float learning_rate, float decay)->void;
	auto AddLayer(string type, int num_hidden, string activation)->int;
	auto Train(float* X, int* Y, int training_iteration, string cost_function)->void;
	auto Validate(float* X, int* Y, int num_test)->float;
	auto Classify(float* X, int* Y, int num_test)->void;


private:

	// the structure that stores attrebutes of each dense layer as 
	// well as weight, bias, input and output vectors for each layer
	// memory is assigned dinamically once a layer is added to
	// the model, and released by the destructor

	struct Layer
	{
		string type;
		string activation;
		string weight_initializer;
		int num_input;
		int num_hidden;
		int num_output;

		float* weight = nullptr;
		float* bias = nullptr;
		float* input = nullptr;
		float* output = nullptr;

		Layer(string t_type, string t_activation, string t_weight_initializer,
			int t_num_input, int t_num_hidden, int t_num_output)
		{
			type = t_type;
			activation = t_activation;
			weight_initializer = t_weight_initializer;
			num_input = t_num_input;
			num_hidden = t_num_hidden;
			num_output = t_num_output;

			if ("Dense" == type)
			{
				weight = new float[num_hidden * num_input]; // stored row-wise
				bias = new float[num_hidden];
				input = new float[num_hidden * num_output]; // (linear output of the layer) stored column-wise
				output = new float[num_hidden * num_output]; // (activated output of the layer) stored column-wise

				if ("random" == weight_initializer) // randomly initialize weghts 
				{
					float tmp_limit = 0.05f;
					for (int idx1 = 0; idx1 < num_hidden; idx1++)
					{
						bias[idx1] = 0;
						for (int idx2 = 0; idx2 < num_input; idx2++)
						{
							weight[idx1 * num_input + idx2] =
								-tmp_limit + 2 * tmp_limit * (rand() / float(RAND_MAX));
						}
					}
				}
			}
		}

		Layer(Layer& other) = delete;

		Layer(Layer&& other)
		{
			type = other.type;
			activation = other.activation;
			weight_initializer = other.weight_initializer;
			num_input = other.num_input;
			num_hidden = other.num_hidden;
			num_output = other.num_output;
			weight = move(other.weight); other.weight = nullptr;
			bias = move(other.bias); other.bias = nullptr;
			input = move(other.input); other.input = nullptr;
			output = move(other.output); other.output = nullptr;
			//cout << "Layer move is called" << endl;
		}
	};


	// the structure that stores gradients of backpropagation
	// memory is assigned dinamically once a layer is added to
	// the model, and released by the destructor
	struct Grad
	{
		string type;
		int num_input;
		int num_hidden;
		int num_output;

		float* dev_input = nullptr;
		float* dev_weight = nullptr;
		float* dev_bias = nullptr;
		float* dev_output = nullptr;

		Grad(string t_type,
			int t_num_input,
			int t_num_hidden,
			int t_num_output)
		{
			type = t_type;
			num_input = t_num_input;
			num_hidden = t_num_hidden;
			num_output = t_num_output;

			if ("Dense" == type)
			{
				dev_input = new float[num_hidden * num_output];
				dev_weight = new float[num_hidden * num_input];
				dev_bias = new float[num_hidden];
				dev_output = new float[num_input * num_output];
			}
		}

		Grad(Grad& other) = delete;

		Grad(Grad&& other)
		{
			type = other.type;
			num_input = other.num_input;
			num_hidden = other.num_hidden;
			num_output = other.num_output;
			dev_weight = move(other.dev_weight); other.dev_weight = nullptr;
			dev_bias = move(other.dev_bias); other.dev_bias = nullptr;
			dev_input = move(other.dev_input); other.dev_input = nullptr;
			dev_output = move(other.dev_output); other.dev_output = nullptr;
			//cout << "Grad move is called" << endl;
		}
	};




	int num_feature_;
	int num_train_;
	int num_output_;
	int count_layers_;
	int num_test_; // number of test samples during testing
	bool IsTest_; // indicates if the model is used for training or testing
				  // false for training, true for testing
	int cost_save_interval_;

	vector<Layer> layers_; // stores each added layer to the model
	vector<Grad> grads_;   // stores gradients of each layer

	string cost_function_; // type of cost function: either cross_entropy or mean_square
	string optimizer_;     // gradient descent
	float learning_rate_;
	float decay_;
	int training_iteration_;
	string weight_initializer_;

	auto dev_relu(float val)->float;
	auto dev_sigmoid(float val)->float;
	auto ArgMax(int* Y_pred)->void;
	auto CalcActivation(int layer, float* X)->void;
	auto Activate(int layer)->void;
	auto FeedForward(float* X, int* Y, int iter)->void;
	auto BackPropagate(float* X, int* Y)->void;
	auto UpdateWeights()->void;
};





#endif