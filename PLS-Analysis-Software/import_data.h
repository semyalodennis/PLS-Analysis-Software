/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#pragma once

#include <fstream>
#include <armadillo>

using namespace arma;

// Import_data from text file
// filename The name of the text file
// spectra_data x data
// response_data y (reference data) data

bool Import_Data(const std::string& filename, arma::mat& spectra_data, arma::vec& response_data)
{
    std::ifstream infile(filename);

    arma::mat tempor;
   
    bool Loaded_file = tempor.load(infile);
    if (Loaded_file) {
        spectra_data = tempor.cols(1, tempor.n_cols - 1);
        response_data = tempor.col(0);

    }
    return Loaded_file;
}
