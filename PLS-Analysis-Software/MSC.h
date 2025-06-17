/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#pragma once
#include<mlpack/core.hpp>
#include<mlpack/methods/linear_regression/linear_regression.hpp>

using namespace mlpack;
using namespace arma;

void MSC_Process(mat X, arma::mat& MSC_X)
{
    //Multiplicative Scatter/Signal Correction - interpretation from matlab code
    //Xmsc = (xorig - a)/b. a is the intercept, b is the coefficients. a & b obtained from multiple linear regression
    //Mean of each row in input
    //SpectralMean will act as the independent/x/predictor variable for the linear regression
    //SpectralMean = predictor
    //mat SpectralMean = mean(input, 1);
    mat predictor_col = mean(X, 1);
    //cout << "predictor_col data size: " << size(predictor_col) << endl;
    mat predictor = trans(predictor_col);
    //cout << "predictor data size: " << size(predictor) << endl;

    //Create a matrix to store MSC corrected Values
    MSC_X = zeros<mat>(X.n_rows, X.n_cols);

    for (int i = 0; i < X.n_cols; i++)
    {
        //Each column will act as the dependent/response/y variable for the linear regression
        vec response_col = X.col(i);
        //cout << "response_col data size: " << size(response_col) << endl;
        rowvec response = trans(response_col);
        //cout << "response data size: " << size(response) << endl;

        //Perform linear regression on predictor and response
        regression::LinearRegression regressor;
        regressor.Train(predictor, response);

        //Obtain the coefficients and intercept
        arma::vec parameters = regressor.Parameters();
        //cout << "coefficients data size: " << size(parameters) << endl;
        //parameters.print("coefficients");

        //Intercept
        //double a = as_scalar(parameters(0));
        double a = parameters(0);
        //cout << "Intercept: " << a << endl;

        //Coefficient
        //double b = as_scalar(parameters(1));
        double b = parameters(1);
        //cout << "coefficient: " << b << endl;

        //Perform MSC
        MSC_X(span::all, i) = (X(span::all, i) - a) / b;

        //cout << "MSC data size: " << size(XMSC) << endl;
        //XMSC.print("MSC preprocessed data:");

    }
}
