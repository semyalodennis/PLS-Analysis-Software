/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#pragma once
#include <armadillo>

//sgolay Find a arma::matrix of Savitzky-Golay convolution coefficients
//paramters: poly_order, window_size, deriv_order, scaling_factor
// Output A arma::matrix of size window_size by window_size containing Savitzky-Golay coefficients.
// Finds a arma::matrix of Savitzky-Golay convolution coefficients, similar to the
// sgolay function in the Octave and arma::matLAB signal packages. 
//The central row is the filter used for most of the dataset. 
// The outer rows are used on the edges of the signal to be filtered.
arma::mat sgolay(arma::uword poly_order, arma::uword window_size, arma::uword deriv_order, arma::uword scaling_factor)
{
    if (window_size % 2 == 0) {
        std::cerr << "sgolay: invalid window size. Window size must be odd." << std::endl;
        std::cerr << "using window size poly_order + 2 or nearest odd value" << std::endl;
        window_size = poly_order + 3 - (poly_order % 2);
    }

    if (deriv_order > poly_order) {
        std::cerr << "sgolay: derivative order greater than polynomial order." << std::endl;
        std::cerr << "using derivative order poly order - 1" << std::endl;
        deriv_order = poly_order - 1;
    }
    arma::mat F = arma::zeros(window_size, window_size);
    arma::uword k = (window_size - 1) / 2;
    arma::mat C(window_size, poly_order + 1);
    arma::sword signed_window_size = window_size;
    arma::vec column;
    arma::sword row; //starting at 1 not 0
    arma::mat A;
    for (arma::uword i = 0; i <= k; ++i) {
        row = i + 1;
        column = arma::linspace<arma::vec>((1 - row), (signed_window_size - row), window_size);
        for (arma::uword j = 0; j <= poly_order; ++j) {
            C.col(j) = arma::pow(column, j);
        }
        A = pinv(C);
        F.row(i) = A.row(deriv_order);
    }
    arma::sword sign = (deriv_order % 2 == 0 ? 1 : -1);
    arma::uword start_index = 0;
    for (arma::uword i = window_size - 1; i > k; --i) {
        F.row(i) = sign * F.row(start_index);
        start_index++;
    }
    double product = (deriv_order == 0 ? 1 : prod(arma::linspace<arma::vec>(1, deriv_order, deriv_order)));
    double power = pow(scaling_factor, deriv_order);
    F /= (power / product);
    return F;
}

//Overview: ApplyFilter Apply FIR filters to a column vector.
//x Data to be filtered
//coefficients A arma::matrix of FIR filters.
//window_size Filter window size
//Output: Filtered data
//Apply FIR filters to a column vector. 
//The central row of coefficients contains the filter used for most of the data. 
//The first (window_size - 1)/2 rows are filtered with the lower rows ofccoefficients, and likewise for the
//last (window_size - 1)/2 rows and the first rows of coefficients.
arma::vec ApplyFilter_SG(const arma::vec& x, arma::mat coefficients, arma::uword window_size)
{
    arma::uword k = (window_size - 1) / 2;
    arma::vec filter = trans(coefficients.row(k));
    //coefficients output by sgolay are in reverse order expected.
    filter = fliplr(filter);
    arma::vec filtered_data = arma::conv(x, filter, "same");
    //Now we have to fix the ends of the filtered data
    filtered_data.rows(0, k) = coefficients.rows(k, window_size - 1) * x.rows(0, window_size - 1);
    filtered_data.rows(x.n_rows - k, x.n_rows - 1) = coefficients.rows(0, k - 1) * x.rows(x.n_rows - window_size, x.n_rows - 1);
    return filtered_data;
}

//Overview - sgolayfilt Apply Savitzky-Golay smoothing to each column of a arma::matrix
//x Input arma::matrix. Each column is a signal
//poly_order Polynomial order for smoothing
//window_size Size of filter window. Must be odd and larger than poly order
//deriv_order Derivative order, to extract derivatized data directly
//scaling_factor A scaling factor
//Output Smoothed data
arma::mat sgolayfilt(const arma::mat& x, arma::uword poly_order, arma::uword window_size, arma::uword deriv_order, arma::uword scaling_factor)
{
    arma::mat return_value(x.n_rows, x.n_cols);

    //this function filters by column.
    //any invalid window size is set to preferred poly_order + 2 or poly_order + 3 for even poly_order
    if ((poly_order > window_size) || (window_size % 2 == 0)) {
        window_size = poly_order + 3 - (poly_order % 2);
    }
    //if deriv_order is too high, make it one less than polynomial order
    if (deriv_order > poly_order) {
        deriv_order = poly_order - 1;
    }
    if (x.n_rows < window_size) {
        std::cerr << "sgolayfilt: not enough data for filter window of this size" << std::endl;
        return x;
    }

    arma::mat coefficients = sgolay(poly_order, window_size, deriv_order, scaling_factor);

    for (arma::uword i = 0; i < x.n_cols; ++i)
        return_value.col(i) = ApplyFilter_SG(x.col(i), coefficients, window_size);

    return return_value;
}
