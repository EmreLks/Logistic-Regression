// author: emrelks

#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
using namespace std;

void InitialWeightAndBias(double *weight, int weightCount, double &bias);
double Sigmoid(double z);
double LossFunction(double expResult, double actResult);
void ReadDataFromFile(string fileName, double *data);
void VectorMatrixMultiplication(int w, int h, double *i_matrix, double *i_vector, double *o_vector, double bias = 0);
double ForwardBackwardPropagation(double *x, double *y, double *w, double &bias);
void MatrixTranpose(int w, int h, double *i_matrix, double *o_matrix);
void Test(double *x, double *y, double *w, double bias);

const double LRate = 0.01;
const double Threshold = 0.001;
const int FeatureCount = 30;
const int TrainCount = 455;
const int TestCount = 114;
const int MaxIteration = 1000;


int main()
{
	string trainXPath = "data/x_train.txt", trainYPath = "data/y_train.txt", testXPath = "data/x_test.txt", testYPath = "data/y_test.txt";


	double x_Train[FeatureCount * TrainCount], x_Test[FeatureCount * TestCount], y_Train[TrainCount], y_Test[TestCount];
	double weight[FeatureCount];
	double bias;
	double cost = 0;

	// Read data from file.
	ReadDataFromFile(trainXPath, x_Train);
	ReadDataFromFile(testXPath, x_Test);
	ReadDataFromFile(trainYPath, y_Train);
	ReadDataFromFile(testYPath, y_Test);

	InitialWeightAndBias(weight, FeatureCount, bias);

	// Train.
	for (int i = 0; i < MaxIteration; i++)
	{
		cost = ForwardBackwardPropagation(x_Train, y_Train, weight, bias);

		if (i % 10 == 0)
		{
			cout << "Cost after iteration " << i <<" - " <<cost << endl;
		}
		// else do nothing.
	}
	// end of the for loop.

	// Test.
	Test(x_Test, y_Test, weight, bias);


	getchar();
	return 0;
}

double LossFunction(double expResult, double actResult)
{
	double result = 0;

	if (expResult == 0)
	{
		result = -log(1 - actResult);
	}
	else if (expResult == 1)
	{
		result = -log(actResult);
	}

	return result;
}

void InitialWeightAndBias(double *weight, int weightCount, double &bias)
{
	bias = 0;
	for (int i = 0; i < weightCount; i++)
	{
		weight[i] = 0.01;
	}
	// end of the for loop.
}
double Sigmoid(double z)
{
	double y = 0;

	y = 1 / (1 + exp(-z));

	return y;
}
void ReadDataFromFile(string fileName, double *data)
{
	ifstream infile;
	infile.open(fileName);

	if (infile.is_open() == false)
	{
		cout << fileName << " could not be opened" << endl;
		return;
	}

	long long index = 0;

	while (!infile.eof())
	{
		infile >> data[index];
		index++;
	}
	// End of the loop.
	infile.close();

	cout << fileName << " successfully read" << endl;
}

void VectorMatrixMultiplication(int w, int h, double *i_matrix, double *i_vector, double *o_vector, double bias)
{
	for (int i = 0; i < h; i++)
	{
		o_vector[i] = 0.0;

		for (int j = 0; j < w; j++)
		{
			o_vector[i] += i_vector[j] * i_matrix[(i * w) + j] + bias;
		}
		// End of the loop.
	}
	// End of the loop.
}

void MatrixTranpose(int w, int h, double *i_matrix, double *o_matrix)
{
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			o_matrix[(j * h) + i] = i_matrix[(i * w) + j];
		}
		// End of the loop.
	}
	// End of the loop.
}

double ForwardBackwardPropagation(double *x, double *y, double *w, double &bias)
{
	double z[TrainCount];
	double yResult[TrainCount];
	double lossResult = 0;
	double costResult = 0;
	double ActExpectedDifferance[TrainCount];
	double sumDifferance = 0;
	double derivativeWeight[FeatureCount];
	// Z = X.W + b;
	VectorMatrixMultiplication(FeatureCount, TrainCount, x, w, z, bias);

	// Apply Sigmoid Function.
	// Find loss result and cost result.
	// yResult = 1 / (1 + e^(-x))
	for (int i = 0; i < TrainCount; i++)
	{
		yResult[i] = Sigmoid(z[i]);

		ActExpectedDifferance[i] = yResult[i] - y[i];

		sumDifferance = sumDifferance + ActExpectedDifferance[i];

		// Expected Result - Actual Result.
		lossResult = LossFunction(y[i], yResult[i]);

		costResult += lossResult;
	}
	// else do nothing.

	costResult = costResult / TrainCount; // for scaling.

	// derivative weight.
	// (actual - expected) * X^t.
	double transposeX[TrainCount * FeatureCount];

	// Transpose.
	MatrixTranpose(FeatureCount, TrainCount, x, transposeX);

	// get derivative weight.
	VectorMatrixMultiplication(TrainCount, FeatureCount, transposeX, ActExpectedDifferance, derivativeWeight, 0);

	// Update weight.
	for (int i = 0; i < FeatureCount; i++)
	{
		w[i] = w[i] - LRate * derivativeWeight[i];
	}
	// Update bias.
	bias = bias - LRate * (sumDifferance / TrainCount);

	return costResult;
}
void Test(double *x, double *y, double *w, double bias)
{
	double z[TestCount];
	double yResult[TestCount];
	double accuracy = 0;
	// Z = X.W + b;
	VectorMatrixMultiplication(FeatureCount, TestCount, x, w, z, bias);

	// Apply Sigmoid Function.
	// Find loss result and cost result.
	// yResult = 1 / (1 + e^(-x))
	for (int i = 0; i < TestCount; i++)
	{
		yResult[i] = Sigmoid(z[i]);

		if (yResult[i] < 0.5)
		{
			yResult[i] = 0;
		}
		else
		{
			yResult[i] = 1;
		}

		accuracy = accuracy + abs(y[i] - yResult[i]);
	}
	// else do nothing.

	accuracy = 100 - (accuracy / TestCount) * 100;

	cout << "Test accuracy: " << accuracy << endl;
}