/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#pragma once

/**
 * A simple Standard Scaler class
 *
 * Given an input dataset this class helps you to Standardize features
 * by removing the mean and scaling to unit variance.
 *
 * \[z = (x - u) / s\]
 *
 * where u is the mean of the training samples and s is the standard deviation
 * of the training samples
 * */

#include <iostream>

#include <mlpack/core.hpp>
#include <mlpack\core\data\scaler_methods\standard_scaler.hpp> //D:\vcpkg\installed\x64-windows\include\mlpack\core\data\scaler_methods\standard_scaler.hpp

using namespace mlpack::data;
using namespace mlpack;

void Standardization(arma::mat X, arma::mat& Standardized_data)
{
	// Fit the features
	StandardScaler scale;
	scale.Fit(X);

	// Scale the features
	arma::mat output;
	scale.Transform(X, output);
	//std::cout << "output data size: " << size(output) << std::endl;
	//output.print("output data:");

	//Print Standardized normalized spectra for sample 1
	/*for (int i = 0; i < 1; i++)
	{
		//cout << "Standardized Spectra " << i + 1 << endl;

		//cout << "size of an observation of Standardized Spectra: " << size(output.col(i)) << endl;
		output.col(i).print("Standardized Spectra");

	}

	// Retransform the input
	arma::mat input_retransformed;
	scale.InverseTransform(output, input_retransformed);
	//cout << "input_retransformed data size: " << size(input_retransformed) << endl;*/
	//input_retransformed.print("input_retransformed data:");

	//Transpose standardized matrix to obtain original orientation of input data
	Standardized_data = trans(output);


}
