/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#pragma once

#include <armadillo>

using namespace arma;

// plsregression PLS Regression Using SIMPLS algorithm.
// This is implementation of plsregress from the Octave statistics package and Matlab.

bool plsregression(arma::mat X, arma::mat Y, int components, arma::mat& X_loadings, arma::mat& Y_loadings, arma::mat& X_scores,
                arma::mat& Y_scores, arma::mat& Beta, arma::mat& percent_var, arma::mat& Predicted, arma::mat& MSE ,double &rsquaredPLS,
                double& RMSE)
{    
    uword observations = X.n_rows;
    uword predictors = X.n_cols;
    uword responses = Y.n_cols;

    //Mean centering
    mat Xmeans = arma::mean(X);
    mat Ymeans = arma::mean(Y);

    //Octave does below with bsxfun. After optimization, this should hopefully not be slower.
    //element-wise subtraction of mean
    X.each_row() -= Xmeans; 
    Y.each_row() -= Ymeans;

    mat S = trans(X) * Y;
    mat R = zeros(predictors, components);
    mat P = R;
    mat V = R;
    mat T = zeros(observations, components);
    mat U = T;
    mat Q = zeros(responses, components);
    mat eigvec;
    vec eigval;

    mat q;
    mat r;
    mat t;
    mat nt;
    mat p;
    mat u;
    mat v;
    double max_eigval;
    for (int i = 0; i < components; ++i) {
        eig_sym(eigval, eigvec, (trans(S) * S));
        max_eigval = eigval.max();
        uvec dom_index = find(eigval >= max_eigval);
        uword dominant_index = dom_index(0);

        q = eigvec.col(dominant_index);

        //X block factor weights
        r = S * q; 
        //X block factor weights
        t = X * r; 
        //center t
        t.each_row() -= mean(t); 
        //compute norm (is arma::norm() the same?)
        nt = arma::sqrt(t.t() * t); 
        //normalize
        t.each_row() /= nt;
        r.each_row() /= nt;

        //X block factor loadings
        p = X.t() * t; 
        //Y block factor loadings
        q = Y.t() * t; 
        //Y block factor scores
        u = Y * q; 
        v = p;

        //Ensure orthogonality
        if (i > 0) {
            v = v - V * (V.t() * p);
            u = u - T * (T.t() * u);
        }
        //normalize orthogonal loadings
        v.each_row() /= arma::sqrt(trans(v) * v); 
        //deflate S wrt loadings
        S = S - v * (trans(v) * S); 
        R.col(i) = r;
        T.col(i) = t;
        P.col(i) = p;
        Q.col(i) = q;
        U.col(i) = u;
        V.col(i) = v;
    }

    //Regression coefficients
    mat B = R * trans(Q);
    Predicted = T * trans(Q);
    Predicted.each_row() += Ymeans;

    //Octave creates copies from inputs before sending to output. Doing same
    //here just to be safe.
    mat Beta1 = B;
    X_scores = T;
    X_loadings = P;
    Y_scores = U;
    Y_loadings = Q;
    //projection = R;

    //Calculate intercept and include it in the beta coefficient
    mat intercept = Ymeans - (Xmeans * Beta1);
    Beta = join_vert(intercept, Beta1);

    //2-by-ncomp matrix PCTVAR containing the percentage of variance explained by the model. 
    //The first row of PCTVAR contains the percentage of variance explained in X by each PLS component, 
    //and the second row contains the percentage of variance explained in Y.
    percent_var.set_size(2, P.n_cols);
    //Percentage variance in X (predictor variable)
    percent_var.row(0) = sum(arma::abs(P) % arma::abs(P)) / sum(sum(arma::abs(X) % arma::abs(X)));

    //Percentage variance in Y (Response variable)
    percent_var.row(1) = sum(arma::abs(Q) % arma::abs(Q)) / sum(sum(arma::abs(Y) % arma::abs(Y)));

    //From Matlab (plsregress function source code)
    //Compute MSE for models with 0:ncomp PLS components
    //Mean Squared Error (MSE)
    //2-by-(NCOMP+1) matrix MSE containing estimated mean-squared errors for PLS models with 0:ncomp components. 
    //The first row of MSE contains mean-squared errors for the predictor variables in X, 
    //and the second row contains mean-squared errors for the response variable(s) in Y.
    //mse = zeros(2,ncomp+1,class(pctVar)); 
    MSE.set_size(2, P.n_cols+1);
    //cout << "Size of MSE" << size(MSE) << endl;

    //mse(1,1) = sum(sum(abs(X0).^2, 2));
    MSE(0, 0) = sum(sum(arma::abs(X) % arma::abs(X),1));
    //mse(2,1) = sum(sum(abs(Y0).^2, 2));
    MSE(1, 0) = sum(sum(arma::abs(Y) % arma::abs(Y), 1));

    mat Xreconstructed;
    mat Yreconstructed;
    for (int i = 0; i < components; ++i)
    {
        Xreconstructed = X_scores.cols(span(0, i)) * X_loadings.cols(span(0, i)).t();
        Yreconstructed = X_scores.cols(span(0, i)) * Y_loadings.cols(span(0, i)).t();

        MSE(0, i+1) = sum(sum(arma::abs(X - Xreconstructed) % arma::abs(X - Xreconstructed), 1));
        MSE(1, i+1) = sum(sum(arma::abs(Y - Yreconstructed) % arma::abs(Y - Yreconstructed), 1));
    }

    uword Number_Predictions = Y.n_rows;
    MSE /= Number_Predictions;

    //Gaining back original Y before calculation of r square
    Y.each_row() += Ymeans;
    //cout << "Y : " << Y << endl;
    //r-squared value
    /*TSS = sum((y - mean(y)). ^ 2);
    RSS_PLS = sum((y - yfitPLS). ^ 2);
    rsquaredPLS = 1 - RSS_PLS / TSS*/

    double TSS = accu((Y - as_scalar(mean(Y))) % (Y - as_scalar(mean(Y))));
    //cout << "TSS : " << TSS << endl;
    //mat residual_squared = square(Y - fitted);
    //cout << "residual_squared : " <<endl<< residual_squared << endl;
    double RSS = accu(square(Y - Predicted)); // find the overall sum regardless of object type
    //cout << "RSS : " << RSS << endl;
    rsquaredPLS = 1 - (RSS / TSS);
    
    //Root mean squared error
    //Note - I have used RSS because it has a similar expression to a section RMSE formula
    RMSE = sqrt(RSS /Number_Predictions);


    return true;

}
