/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#pragma once

#include <armadillo>

//Standard Normal Variate preprocessing method coded with reference to 
//Introduction to multivariate statistical analysis in chemometrics

using namespace arma;

//mat Standard_Normal_variate(mat X, arma::mat& SNV_X)
void Standard_Normal_variate(mat X, arma::mat& SNV_X)
{
	//Calculate the mean of each row. 
	//mean( M, dim )- For matrix M, find the statistic for each column (dim=0), or each row (dim=1)
	vec Mean_rowise = mean(X, 1);

	//Calculate the standard deviation of each row. 
	//stddev(M, norm_type, dim)-The norm_type argument is optional; by default norm_type=0 is used
	//the default norm_type = 0 performs normalisation using N - 1 (where N is the number of samples), 
	//providing the best unbiased estimator
	//using norm_type = 1 performs normalisation using N, which provides the second moment around the mean
	vec StdDev_rowise = stddev(X, 0, 1);

	//Subtract Mean_rowise from each column of X
	//X.each_col() -= Mean_rowise;
	mat X_CenterROW = X.each_col() - Mean_rowise;


	//Divide the centered X (each column of X) by StdDev_rowise
	//X.each_col() /= Mean_rowise;
	SNV_X = X_CenterROW.each_col() / StdDev_rowise;
	//The output X gives the SNV normalized matrix
	//SNV_X = X;
}
