/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#pragma once

/*
https://cppsecrets.com/users/489510710111510497118107979811497495464103109971051084699111109/C00-MLPACK-MeanNormalization.php
Introduction:

MeanNormalization is a simple class, in MLpack, which scales the dataset according it its overall mean value.

Formula for Mean Normalization is given below :

{\displaystyle x'={\frac {x-{\text{average}}(x)}{{\text{max}}(x)-{\text{min}}(x)}}} - [z = x - average(x) / (max(x) - min(x))\]

where, x is the orginal valueand x' is the normalized value.*/

#include <mlpack/core.hpp>
#include <mlpack\core\data\scaler_methods\mean_normalization.hpp> //D:\vcpkg\installed\x64-windows\include\mlpack\core\data\scaler_methods

using namespace mlpack::data;
using namespace mlpack;
using namespace arma;

void Mean_Normalize(mat X, arma::mat& Mean_Normalized_data)
{
	// Fit the features
	MeanNormalization scale;
	scale.Fit(X);

	// Scale the features
	arma::mat output;
	scale.Transform(X, output);
	//cout << "output data size: " << size(output) << endl;
	//output.print("output data:");

	//Print mean normalized spectra for sample 1
	/*for (int i = 0; i < 1; i++)
	{
		cout << "Normalized Spectra " << i + 1 << endl;

		cout << "size of an observation of Normalized Spectra: " << size(output.col(i)) << endl;
		output.col(i).print("Normalized Spectra");

	}*/

	// Retransform the input
	//arma::mat input_retransformed;
	//scale.InverseTransform(output, input_retransformed);
	//cout << "input_retransformed data size: " << size(input_retransformed) << endl;
	//input_retransformed.print("input_retransformed data:");

	//Transpose mean normalized matrix to obtain original orientation of input data
	Mean_Normalized_data = trans(output);
	
}
