/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#include "mainwindow.h"

#include <QApplication>

//Qt
#include <QSplashScreen>
#include <QTimer>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QSplashScreen *splash = new QSplashScreen;
    splash->setPixmap(QPixmap(":/Res/Icons/CNU_logo.png"));
    splash->show();

    MainWindow w;

     QTimer::singleShot(2500, splash, SLOT(close()));
     QTimer::singleShot(2500, &w, SLOT(show()));


    //w.show();
    return a.exec();
}
