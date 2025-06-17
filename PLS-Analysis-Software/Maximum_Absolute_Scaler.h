/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#pragma once

/*https://cppsecrets.com/users/489510710111510497118107979811497495464103109971051084699111109/C00-MLPACK-MaxAbsScaler.php
AbsMaxScaler in MLpack :

The MaxAbsScaler automatically scales the data to a[-1, 1] range based on the absolute maximum.
\[z = x / max(abs(x))\]
*
*where max(abs(x)) is maximum absolute value of feature.*/

#include <mlpack/core.hpp>
#include <mlpack\core\data\scaler_methods\max_abs_scaler.hpp> //D:\vcpkg\installed\x64-windows\include\mlpack\core\data\max_abs_scaler.hpp

//using namespace std;
using namespace mlpack::data;
using namespace mlpack;

void Max_Absolute_Norm(arma::mat X, arma::mat& Max_Abs_Normalized)
{
	// Fit the features
	MaxAbsScaler scale;
	scale.Fit(X);

	// Scale the features
	arma::mat output;
	scale.Transform(X, output);
	//std::cout << "output data size: " << size(output) << std::endl;
	//output.print("output data:");

	//Print max absolute normalized spectra for sample 1
	/*for (int i = 0; i < 1; i++)
	{
		cout << "Max Abs Normalized Spectra " << i + 1 << endl;

		cout << "size of an observation of Max Abs Normalized Spectra: " << size(output.col(i)) << endl;
		output.col(i).print("Max Abs Normalized Spectra");

	}

	// Retransform the input
	arma::mat input_retransformed;
	scale.InverseTransform(output, input_retransformed);
	cout << "input_retransformed data size: " << size(input_retransformed) << endl;*/
	//input_retransformed.print("input_retransformed data:");

	//Transpose Max abs normalized matrix to obtain original orientation of input data
	Max_Abs_Normalized = trans(output);

}
