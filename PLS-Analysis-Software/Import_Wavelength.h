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

bool Import_Wavelength(const std::string& filename, arma::vec& Wavelength)
{
    std::ifstream infile(filename);

    arma::mat tempor;
   
    bool Loaded_file = tempor.load(infile);
    if (Loaded_file)
    {
        Wavelength = tempor.col(0);
    }

    return Loaded_file;
}
