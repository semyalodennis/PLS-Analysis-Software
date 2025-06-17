/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#include "plt_spectra.h"
#include "ui_plt_spectra.h"

Plt_Spectra::Plt_Spectra(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Plt_Spectra)
{
    ui->setupUi(this);

}

Plt_Spectra::~Plt_Spectra()
{
    delete ui;

}


