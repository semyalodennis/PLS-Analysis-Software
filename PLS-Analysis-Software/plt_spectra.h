/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#ifndef PLT_SPECTRA_H
#define PLT_SPECTRA_H

#include <QDialog>

namespace Ui {
class Plt_Spectra;
}

class Plt_Spectra : public QDialog
{
    Q_OBJECT

public:
    explicit Plt_Spectra(QWidget *parent = nullptr);
    ~Plt_Spectra();

public:
    Ui::Plt_Spectra *ui;

//private:
    //Ui::Plt_Spectra *ui;

};

#endif // PLT_SPECTRA_H
