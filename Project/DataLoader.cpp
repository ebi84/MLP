#include "Header.h"

// reading data from the csv file
void LoadData(float* X, int* Y, int num_feature, int num_train)
{
	ifstream file;

	file.open("digits_1.csv");

	if (file.is_open())
	{
		for (int idx1 = 0; idx1 < num_train; idx1++)
		{
			for (int idx2 = 0; idx2 <= num_feature; idx2++)
			{
				if (idx2 < num_feature)
				{
					file >> X[idx1 * num_feature + idx2];
					file.get(); 
				}
				else
				{
					file >> Y[idx1]; 
					file.get(); 
				}
			}
		}
	}
	file.close();
	file.clear();
}



void Normalize(float* X, int num_feature, int num_train)
{
	float tmp_min, tmp_max;
	for (int idx1 = 0; idx1 < num_feature; idx1++)
	{
		tmp_min = FLT_MAX; tmp_max = FLT_MIN;
		for (int idx2 = 0; idx2 < num_train; idx2++)
		{
			tmp_min = MIN(tmp_min, X[idx1 + idx2 * num_feature]);
			tmp_max = MAX(tmp_max, X[idx1 + idx2 * num_feature]);
		}
		for (int idx2 = 0; idx2 < num_train; idx2++)
		{
			X[idx1 + idx2 * num_feature] = (X[idx1 + idx2 * num_feature] - tmp_min) / (tmp_max - tmp_min);
		}
	}
}