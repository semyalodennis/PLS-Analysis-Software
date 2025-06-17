/*This file is part of the cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS*/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "qcustomplot.h"

#include <plt_spectra.h> //When object is stored on heap (pointer)
#include "ui_plt_spectra.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    //Train PLSR & PLS-DA Model
    void on_actionTrain_PLSR_Model_triggered();

    void on_PB_importText_clicked();

    void on_RB_Kfold_clicked();

    void on_SP_NSegment_valueChanged(int arg1);

    void on_PB_Train_clicked();

    void on_PB_ViewPlots_clicked();

    void graphClickedPred(QCPAbstractPlottable* plottable, int dataIndex, QMouseEvent* event);

    void graphClickedPredNew(QCPAbstractPlottable* plottable, int dataIndex, QMouseEvent* event);

    void on_actionExit_triggered();

    void on_PB_PlotPred_Ref_clicked();

    void on_PB_Plot_Beta_clicked();

    //void on_PB_Plot_EVar_clicked();

    //Predict/Classify New Dataset using Trained Model

    void on_actionPredict_New_Dataset_triggered();

    void on_PB_Data_2_Pred_clicked();

    void on_PB_Predict_clicked();

    void on_PB_ViewPlots_Pred_clicked();

    void on_PB_Plt_PredRef_NewD_clicked();

    //Train PLS-DA Model (Changing displayed words but same model as PLSR)
    void on_actionTrain_PLS_DA_Model_triggered();

    void on_actionClassify_New_Dataset_triggered();

    void on_SP_Classes_valueChanged(int arg1);

    void on_PB_PltPred_Ref_Class_clicked();

    void on_SP_Classes_CLFY_valueChanged(int arg1);

    void on_PB_PltPred_Ref_Class_CLFY_clicked();

    void on_PB_Save_PredRef_clicked();

    void on_PB_Sve_PredRef_Class_clicked();

    void on_PB_Save_Beta_clicked();

    void on_PB_Save_PredRef_NewD_clicked();

    void on_PB_Sve_PredRef_Class_CLFY_clicked();

    void on_actionOpen_triggered();

    void on_RB_Savitzky_clicked();

    void on_RB_Smooth_clicked();

    void on_RB_Savitzky_Pred_clicked();

    void on_RB_Smooth_Pred_clicked();

    void on_PB_PlotSpectra_clicked();

    void on_pushButton_clicked();

    void on_PB_plot_Spectra_Pred_clicked();

    void on_actionAbout_triggered();

private:
    Ui::MainWindow *ui;
    Plt_Spectra * pltspectra;
    //Ui::Plt_Spectra *ui_ptr2;
};
#endif // MAINWINDOW_H
