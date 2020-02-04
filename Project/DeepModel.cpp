#include "Header.h"




DeepModel::DeepModel(int t_num_feature, int t_num_train, int t_num_output)
{
	count_layers_ = 0;

	num_feature_ = t_num_feature;
	num_train_ = t_num_train;
	num_output_ = t_num_output;
	num_test_ = 0;
	IsTest_ = false;

	cost_function_ = "cross_entropy";
	optimizer_ = "GradientDescent";
	learning_rate_ = 0.001f;
	decay_ = 0.0001f;
	training_iteration_ = 5000;
	cost_save_interval_ = 50;

	weight_initializer_ = "random";
}




DeepModel::~DeepModel()
{
	for (int i = 0; i < count_layers_; i++)
	{
		delete[] layers_[i].weight;
		delete[] layers_[i].bias;
		delete[] layers_[i].input;
		delete[] layers_[i].output;

		delete[] grads_[i].dev_weight;
		delete[] grads_[i].dev_bias;
		delete[] grads_[i].dev_input;
		delete[] grads_[i].dev_output;
	}
	//cout << "memory is released!" << endl;
}



// this function adds a dense layer to the model and initializes all attributes and weights
// for Layer and Grad structures
int DeepModel::AddLayer(string t_type, int t_num_hidden, string t_activation)
{
	if ("Dense" != t_type)
	{
		cerr << "Model only supports dense layers" << endl; return -1;
	}

	int tmp_num_input = (0 == count_layers_ ? num_feature_ :
		layers_[count_layers_ - 1].num_hidden);

	layers_.emplace_back(Layer(t_type, t_activation, weight_initializer_,
		tmp_num_input, t_num_hidden, num_train_));

	grads_.emplace_back(Grad(t_type, tmp_num_input, t_num_hidden, num_train_));
	count_layers_++;

	return 0;
}


// this function calculates the input (WX+b) as well as the activated output during feed forward
void DeepModel::CalcActivation(int layer, float* X)
{
	if ("Dense" == layers_[layer].type)
	{
		int dim1 = layers_[layer].num_hidden;
		int dim2 = IsTest_ ? num_test_ : layers_[layer].num_output;
		int dim3 = layers_[layer].num_input;
		int idx1, idx2, idx3;
		float tmp_val;

		if (0 == layer)
		{
#pragma omp parallel for private(idx2, idx3, tmp_val)
			for (idx1 = 0; idx1 < dim1; idx1++)
			{
				for (idx2 = 0; idx2 < dim2; idx2++)
				{
					tmp_val = layers_[layer].bias[idx1];
					for (idx3 = 0; idx3 < dim3; idx3++)
					{
						tmp_val += (layers_[layer].weight[idx1 * dim3 + idx3] * X[idx3 + idx2 * dim3]);
					}
					layers_[layer].input[idx1 * dim2 + idx2] = tmp_val;
				}
			}
		}
		else
		{
#pragma omp parallel for private(idx2, idx3, tmp_val)
			for (idx1 = 0; idx1 < dim1; idx1++)
			{
				for (idx2 = 0; idx2 < dim2; idx2++)
				{
					tmp_val = layers_[layer].bias[idx1];
					for (idx3 = 0; idx3 < dim3; idx3++)
					{
						tmp_val += (layers_[layer].weight[idx1 * dim3 + idx3] *
							layers_[layer - 1].output[idx3 * dim2 + idx2]);
					}
					layers_[layer].input[idx1 * dim2 + idx2] = tmp_val;
				}
			}
		}

		if ("none" != layers_[layer].activation)
		{
			this->Activate(layer);
		}
		else
		{
			int dim3 = dim1 * dim2;
			for (int idx1 = 0; idx1 < dim3; idx1++)
			{
				layers_[layer].output[idx1] = layers_[layer].input[idx1];
			}
		}
	}
}


// this function activates the linear input using different activation functions
void DeepModel::Activate(int layer)
{
	int dim1 = layers_[layer].num_hidden;
	int dim2 = IsTest_ ? num_test_ : layers_[layer].num_output;
	int dim3 = dim1 * dim2;
	int idx1, idx2;

	// relu
	if ("relu" == layers_[layer].activation)
	{
#pragma omp parallel for private(idx1)
		for (idx1 = 0; idx1 < dim3; idx1++)
		{
			layers_[layer].output[idx1] = MAX(layers_[layer].input[idx1], 0);
		}
	}
	// sigmoid
	else if ("sigmoid" == layers_[layer].activation)
	{
#pragma omp parallel for private(idx1)
		for (idx1 = 0; idx1 < dim3; idx1++)
		{
			layers_[layer].output[idx1] = 1 / (1 + exp(-layers_[layer].input[idx1]));
		}
	}
	// softmax. in order to avoid overflow, this module adds -max(a) of 
	// each column to the exponential value.
	else if ("softmax" == layers_[layer].activation)
	{
		float tmp_max, tmp_sum;
#pragma omp parallel for private(idx2, tmp_max, tmp_sum)
		for (idx1 = 0; idx1 < dim2; idx1++)
		{
			tmp_max = FLT_MIN; tmp_sum = 0;
			for (idx2 = 0; idx2 < dim1; idx2++)
			{
				tmp_max = MAX(tmp_max, layers_[layer].input[idx1 + idx2 * dim2]);
			}
			for (idx2 = 0; idx2 < dim1; idx2++)
			{
				layers_[layer].output[idx1 + idx2 * dim2] =
					exp(layers_[layer].input[idx1 + idx2 * dim2] - tmp_max);
			}
			for (idx2 = 0; idx2 < dim1; idx2++)
			{
				tmp_sum += layers_[layer].output[idx1 + idx2 * dim2];
			}
			for (idx2 = 0; idx2 < dim1; idx2++)
			{
				layers_[layer].output[idx1 + idx2 * dim2] /= tmp_sum;
			}
		}
	}
}



void DeepModel::SetCostFunction(string t_type)
{
	cost_function_ = t_type;
}


// calculates cost function in periodic cost_save_interval_ iterations
void DeepModel::CalcCostFunction(int* Y)
{
	float tmp_cost(0);

	if (cost_function_ == "cross_entropy")
	{
		for (int idx1 = 0; idx1 < num_train_; idx1++)
		{
			// #pragma omp parallel for reduction(-:tmp_cost)
			for (int idx2 = 0; idx2 < num_output_; idx2++)
			{
				tmp_cost -= (idx2 == Y[idx1] ? log(layers_[count_layers_ - 1].output[idx2 * num_train_ + idx1]) :
					log(1 - layers_[count_layers_ - 1].output[idx2 * num_train_ + idx1]));

			}
		}
		costs.push_back(tmp_cost / num_train_);
	}
	else if (cost_function_ == "mean_square")
	{
		for (int idx1 = 0; idx1 < num_train_; idx1++)
		{
			// #pragma omp parallel for reduction(+:tmp_cost)
			for (int idx2 = 0; idx2 < num_output_; idx2++)
			{
				tmp_cost += pow(layers_[count_layers_ - 1].output[idx2 * num_train_ + idx1] - (idx2 == Y[idx1]), 2);

			}
		}
		costs.push_back(tmp_cost / (2 * num_train_));
	}
}


// this function performs feed forward during both training and test sessions
void DeepModel::FeedForward(float* X, int* Y, int iter)
{
	for (int layer = 0; layer < count_layers_; layer++)
	{
		this->CalcActivation(layer, X);
	}
	if ((0 == (iter % cost_save_interval_)) || (iter == training_iteration_ - 1)) { this->CalcCostFunction(Y); }
}




void DeepModel::SetOptimizer(string t_optimizer, float t_learning_rate, float t_decay)
{
	optimizer_ = t_optimizer;
	learning_rate_ = t_learning_rate;
	decay_ = t_decay;
}



// backpropagation
void DeepModel::BackPropagate(float* X, int* Y)
{
	int dim1, dim2, dim3;
	float tmp_val;
	int idx1, idx2, idx3;

	for (int layer = count_layers_ - 1; layer >= 0; layer--)
	{
		// calculating dev_input for the last layer
		if (count_layers_ - 1 == layer && "cross_entropy" == cost_function_)
		{
			if ("sigmoid" == layers_[count_layers_ - 1].activation || "softmax" == layers_[count_layers_ - 1].activation)
			{
#pragma omp parallel for private(idx2)
				for (idx1 = 0; idx1 < num_train_; idx1++)
				{
					for (idx2 = 0; idx2 < num_output_; idx2++)
					{
						grads_[count_layers_ - 1].dev_input[idx2 * num_train_ + idx1] =
							layers_[count_layers_ - 1].output[idx2 * num_train_ + idx1] - (idx2 == Y[idx1]);
					}
				}
			}
		}
		else if (count_layers_ - 1 == layer && "mean_square" == cost_function_)
		{
			if ("sigmoid" == layers_[count_layers_ - 1].activation || "softmax" == layers_[count_layers_ - 1].activation)
			{
#pragma omp parallel for private(idx2, tmp_val)
				for (idx1 = 0; idx1 < num_train_; idx1++)
				{
					for (idx2 = 0; idx2 < num_output_; idx2++)
					{
						tmp_val = layers_[count_layers_ - 1].output[idx2 * num_train_ + idx1];
						grads_[count_layers_ - 1].dev_input[idx2 * num_train_ + idx1] =
							(tmp_val - (idx2 == Y[idx1])) * tmp_val * (1 - tmp_val);
					}
				}
			}
		}
		else
		{
			// calculating dev_input for other layers
			dim1 = layers_[layer].num_hidden * layers_[layer].num_output;
			if ("relu" == layers_[layer].activation)
			{
#pragma omp parallel for private(idx1) 
				for (idx1 = 0; idx1 < dim1; idx1++)
				{
					grads_[layer].dev_input[idx1] = grads_[layer + 1].dev_output[idx1] * dev_relu(layers_[layer].input[idx1]);
				}
			}
			else if ("sigmoid" == layers_[layer].activation)
			{
#pragma omp parallel for private(idx1)
				for (idx1 = 0; idx1 < dim1; idx1++)
				{
					grads_[layer].dev_input[idx1] = grads_[layer + 1].dev_output[idx1] * dev_sigmoid(layers_[layer].input[idx1]);
				}
			}
		}


		/* calculating dev_output for all layers except the first layer
			weight'*dev_in */
		if (0 != layer)
		{
			dim1 = layers_[layer].num_input;
			dim2 = layers_[layer].num_output;
			dim3 = layers_[layer].num_hidden;

#pragma omp parallel for private(idx2, idx3, tmp_val)
			for (idx1 = 0; idx1 < dim1; idx1++)
			{
				for (idx2 = 0; idx2 < dim2; idx2++)
				{
					tmp_val = 0;
					for (idx3 = 0; idx3 < dim3; idx3++)
					{
						tmp_val += layers_[layer].weight[idx3 * dim1 + idx1] * grads_[layer].dev_input[idx3 * dim2 + idx2];
					}
					grads_[layer].dev_output[idx1 * dim2 + idx2] = tmp_val;
				}
			}
		}



		/* calculating dev_weight fro all layers:
		dev_weight = dev_input * output_prev' ./ num_train */

		if (0 == layer)
		{
			dim1 = layers_[layer].num_hidden;
			dim2 = num_feature_;
			dim3 = layers_[layer].num_output;

#pragma omp parallel for private(idx2, idx3, tmp_val)
			for (idx1 = 0; idx1 < dim1; idx1++)
			{
				for (idx2 = 0; idx2 < dim2; idx2++)
				{
					tmp_val = 0;
					for (idx3 = 0; idx3 < dim3; idx3++)
					{
						tmp_val += grads_[layer].dev_input[idx1 * dim3 + idx3] * X[idx2 + idx3 * num_feature_];
					}
					grads_[layer].dev_weight[idx1 * dim2 + idx2] = tmp_val / num_train_;
				}
			}
		}
		else
		{
			dim1 = layers_[layer].num_hidden;
			dim2 = layers_[layer].num_input;
			dim3 = layers_[layer].num_output;

#pragma omp parallel for private(idx2, idx3, tmp_val)
			for (idx1 = 0; idx1 < dim1; idx1++)
			{
				for (idx2 = 0; idx2 < dim2; idx2++)
				{
					tmp_val = 0;
					for (idx3 = 0; idx3 < dim3; idx3++)
					{
						tmp_val += grads_[layer].dev_input[idx1 * dim3 + idx3] * layers_[layer - 1].output[idx2 * dim3 + idx3];
					}
					grads_[layer].dev_weight[idx1 * dim2 + idx2] = tmp_val / num_train_;
				}
			}
		}


		// calculating dev_bias = sum(dev_input)/num_train
		dim1 = layers_[layer].num_hidden;
		dim2 = layers_[layer].num_output;

#pragma omp parallel for private(idx2, tmp_val)
		for (idx1 = 0; idx1 < dim1; idx1++)
		{
			tmp_val = 0;
			// #pragma omp parallel for reduction(+:tmp_val)
			for (idx2 = 0; idx2 < dim2; idx2++)
			{
				tmp_val += grads_[layer].dev_input[idx1 * dim2 + idx2];
			}
			grads_[layer].dev_bias[idx1] = tmp_val / num_train_;
		}
	}
}


// deviation of relu
float DeepModel::dev_relu(float val)
{
	return (float)val >= 0;
}


// deviation of sigmoid
float DeepModel::dev_sigmoid(float val)
{
	float tmp_s = 1 / (1 + exp(-val));
	return tmp_s * (1 - tmp_s);
}



// updating weights based on backpropagation and gradient descent
void DeepModel::UpdateWeights()
{
	if ("GradientDescent" == optimizer_)
	{
		int dim1, dim2, idx, layer;

#pragma omp parallel for private(dim1, dim2, idx)
		for (layer = 0; layer < count_layers_; layer++)
		{
			dim1 = layers_[layer].num_hidden;
			dim2 = layers_[layer].num_hidden * layers_[layer].num_input;

			for (idx = 0; idx < dim1; idx++)
			{
				layers_[layer].bias[idx] -= learning_rate_ * grads_[layer].dev_bias[idx];
			}

			for (idx = 0; idx < dim2; idx++)
			{
				layers_[layer].weight[idx] -= learning_rate_ * grads_[layer].dev_weight[idx];
			}
		}
		learning_rate_ *= (1 - decay_);
	}
}



// trains the model using X and Y
void DeepModel::Train(float* X, int* Y, int t_training_iteration, string t_cost_function)
{
	cost_function_ = t_cost_function;
	training_iteration_ = t_training_iteration;

	for (int iter = 0; iter < training_iteration_; iter++)
	{
		this->FeedForward(X, Y, iter);
		if (0 == (iter % cost_save_interval_)) { cout << "cost at iteration " << iter << ": " << costs[costs.size() - 1] << endl; }
		this->BackPropagate(X, Y);
		this->UpdateWeights();
	}
	cout << "final cost: " << costs[costs.size() - 1] << endl;
}



// validates the model for X as input and Y as output
float DeepModel::Validate(float* X, int* Y, int t_num_test)
{
	IsTest_ = true;
	num_test_ = t_num_test;
	int* Y_pred = new int [num_test_];

	for (int layer = 0; layer < count_layers_; layer++)
	{
		this->CalcActivation(layer, X);
	}
	this->ArgMax(Y_pred);

	float accuracy(0);
//#pragma omp parallel for reduction(+:accuracy)
	for (int idx = 0; idx < num_test_; idx++)
	{
		accuracy += (Y[idx] == Y_pred[idx]);
	}
	accuracy /= num_test_;

	IsTest_ = false; // reset test model
	num_test_ = 0;
	delete[] Y_pred;

	return accuracy;
}



// classifies the input X and return the results using the passed Y vector
void DeepModel::Classify(float* X, int* Y, int t_num_test)
{
	IsTest_ = true;
	num_test_ = t_num_test;

	for (int layer = 0; layer < count_layers_; layer++)
	{
		this->CalcActivation(layer, X);
	}
	this->ArgMax(Y);

	IsTest_ = false; // reset test model
	num_test_ = 0;
}



// arg max for classification 
void DeepModel::ArgMax(int* Y_pred)
{
	int tmp_arg, idx1, idx2;
	float tmp_max;

#pragma omp parallel for private(idx2, tmp_arg, tmp_max)
	for (idx1 = 0; idx1 < num_test_; idx1++)
	{
		tmp_arg = 0;
		tmp_max = layers_[count_layers_ - 1].output[idx1];
		for (idx2 = 1; idx2 < num_output_; idx2++)
		{
			if (layers_[count_layers_ - 1].output[idx2 * num_test_ + idx1] > tmp_max)
			{
				tmp_max = layers_[count_layers_ - 1].output[idx2 * num_test_ + idx1];
				tmp_arg = idx2;
			}
		}
		Y_pred[idx1] = tmp_arg;
	}
}