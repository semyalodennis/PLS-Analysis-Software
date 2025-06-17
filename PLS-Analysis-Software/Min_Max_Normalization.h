/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#pragma once

/*Introduction:

Many machine learning(ML) algorithms work better when features are relatively similar scaleand close to normally distributed.

For each value in a feature, MinMaxScaler subtracts the minimum value in the featureand then divides by the range.
The range is the difference between the original maximumand original minimum.

Thus, given an input dataset, the MinMaxScaler class will scale each feature to a given range.
It preserves the shape of the original distribution.It doesn't meaningfully change the information embedded in the original data.

Formula:

X_std = (X - X.min) / (X.max - X.min)

X_scaled = X_std * (max - min) + min

where, min / max : feature_range*/

#include <mlpack/core.hpp>
#include <mlpack\core\data\scaler_methods\min_max_scaler.hpp> //D:\vcpkg\installed\x64-windows\include\mlpack\core\data\scaler_methods\min_max_scaler

using namespace mlpack::data;
using namespace mlpack;
using namespace arma;

void Min_Max_Normalize(mat X, arma::mat& Min_Max_Normalized)
{
	// Fit the features
	MinMaxScaler scale;
	scale.Fit(X);

	// Scale the features
	arma::mat output;
	scale.Transform(X, output);
	//cout << "output data size: " << size(output) << endl;
	//output.print("output data:");

	//Print min-max normalized spectra for sample 1
	/*for (int i = 0; i < 1; i++)
	{
		cout << "min-max Normalized Spectra " << i + 1 << endl;

		cout << "size of an observation of min-max Normalized Spectra: " << size(output.col(i)) << endl;
		output.col(i).print("min-max Normalized Spectra");

	}*/

	// Retransform the input
	/*arma::mat input_retransformed;
	scale.InverseTransform(output, input_retransformed);
	cout << "input_retransformed data size: " << size(input_retransformed) << endl;*/
	//input_retransformed.print("input_retransformed data:");

	//Transpose Min_max normalized matrix to obtain original orientation of input data
	Min_Max_Normalized = trans(output);
}


