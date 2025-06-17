/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#pragma once
#include <armadillo>

//CreateMovingAverageFilter
// window_size The window size of the filter, an odd number representing the width of the window.
// return A moving average filter
// Create a moving average filter of a certain window size.
arma::vec CreateMovingAverageFilter(arma::uword window_size)
{
    double value = 1 / (double)window_size;
    arma::vec filter(window_size);
    filter.fill(value);
    return filter;
}

//Apply the Linear Moving Average Filter
// x The vector to filter
// filter The filter
// return Filtered data 
//Entries near the boundaries of spectra are not processed.
arma::vec ApplyFilter(const arma::vec& x, arma::vec filter)
{
    arma::uword k = (filter.n_elem - 1) / 2;

    //conv( A, B, shape ) - 1D convolution of vectors A and B
    //The shape argument is optional; it is one of:
    //"full" = return the full convolution(default setting), with the size equal to A.n_elem + B.n_elem - 1
    //"same" = return the central part of the convolution, with the same size as vector A
    arma::vec out = conv(x, filter, "same");
    //replace boundaries with values from the original
    out.rows(0, k) = x.rows(0, k);
    out.rows(x.n_elem - k, x.n_elem - 1) = x.rows(x.n_elem - k, x.n_elem - 1);
    return out;
}

