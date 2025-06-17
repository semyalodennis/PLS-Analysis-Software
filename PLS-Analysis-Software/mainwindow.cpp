/*

This is a  cross platform software for Partial Least Squares Descriminant Analysis (PLS-DA)/Classification
and Partial Least Squares Regression Analysis (PLS-R) Summarised as PLS Analysis Software.

DEPARTMENT OF BIOSYSTEMS MACHINERY ENGINEERING
NON DESTRUCTIVE BIOSENSING LABORATORY
CHUNGNAM NATIONAL UNIVERSITY

SUPERVISOR: PROFESSOR BYOUNG-KWAN CHO
DEVELOPER: SEMYALO DENNIS
F
*/


#include "mainwindow.h"
#include "./ui_mainwindow.h"
//#include "ui_plt_spectra.h"

//STL
#include <iostream>
#include <fstream>

//Qt
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>

//Other header files
#include "import_data.h"
#include "plsregress.h"
#include "plsregresscv.h"
#include "qcustomplot.h"
#include "Import_Wavelength.h"

//Armadillo
#include <armadillo>

//Preprocessing Methods
#include "Standard_Normal_Variate.h"
#include "MSC.h"
#include "Savitzky_Golay_method.h"
#include "Moving_Average_Filter.h"
#include "Mean_Normalization.h"
#include "Min_Max_Normalization.h"
#include "Standardization.h"

using namespace std;
using namespace arma;

typedef std::vector<double> stdvec;

//Global Variables
arma::mat spectra_data;
arma::vec response_data;
arma::mat spectra_data_Pred;
arma::vec response_data_Pred;
arma::mat Beta_optm;
arma::mat Predicted_optm;
mat	MSE_trans;
int components;
mat Percent_Variance;
mat predicted_test;
double rsquared_optm;
double RMSE_optm;
QVector<double> Wavelength_qvec;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //Window parameters
    //Title of the application
    setWindowTitle("PLS Analysis Software V1.0");
    setWindowIcon(QIcon(":/Res/Icons/pls.png"));

    ui->stackedWidget->setCurrentIndex(0); //Default widget

    //Make the tooltips of action buttons visible
    ui->menuFile->setToolTipsVisible(true);
    ui->menuTasks->setToolTipsVisible(true);

    //Set icons for home
        //Adding relative path (From resource file)
        QPixmap pic4(":/Res/Icons/Cmake..png");
        QPixmap pic5(":/Res/Icons/CNU_logo.png");
        QPixmap pic6(":/Res/Icons/cpp.png");
        QPixmap pic7(":/Res/Icons/HSILOGO3.png");
        QPixmap pic8(":/Res/Icons/pls.png");
        QPixmap pic10(":/Res/Icons/qt images.png");
        QPixmap pic11(":/Res/Icons/armadillo_logo2.png");

        //Add the icons to the label
        /*ui->LB_Cmake->setPixmap(pic4);
        ui->LB_CNU->setPixmap(pic5);
        ui->LB_cpp->setPixmap(pic6);
        ui->LB_HSI->setPixmap(pic7);
        ui->LB_PLS->setPixmap(pic8);
        ui->LB_Qt->setPixmap(pic10);
        ui->LB_Armadillo->setPixmap(pic11);*/

        //Scaling with width and height of label
        int w4 = ui->LB_Cmake->width();
        int h4 = ui->LB_Cmake->height();
        ui->LB_Cmake->setPixmap(pic4.scaled(w4, h4, Qt::KeepAspectRatio));

        int w5 = ui->LB_CNU->width();
        int h5 = ui->LB_CNU->height();
        ui->LB_CNU->setPixmap(pic5.scaled(w5, h5, Qt::KeepAspectRatio));

        int w6 = ui->LB_cpp->width();
        int h6 = ui->LB_cpp->height();
        ui->LB_cpp->setPixmap(pic6.scaled(w6, h6, Qt::KeepAspectRatio));

        int w7 = ui->LB_HSI->width();
        int h7 = ui->LB_HSI->height();
        ui->LB_HSI->setPixmap(pic7.scaled(w7, h7, Qt::KeepAspectRatio));

        int w8 = ui->LB_PLS->width();
        int h8 = ui->LB_PLS->height();
        ui->LB_PLS->setPixmap(pic8.scaled(w8, h8, Qt::KeepAspectRatio));

        int w10 = ui->LB_Qt->width();
        int h10 = ui->LB_Qt->height();
        ui->LB_Qt->setPixmap(pic10.scaled(w10, h10, Qt::KeepAspectRatio));

        int w11 = ui->LB_Armadillo->width();
        int h11 = ui->LB_Armadillo->height();
        ui->LB_Armadillo->setPixmap(pic11.scaled(w11, h11, Qt::KeepAspectRatio));

    //Connect signal to slot when MSE graph is clicked
    connect(ui->Plot_PredVRef, SIGNAL(plottableClick(QCPAbstractPlottable*, int, QMouseEvent*)), this, SLOT(graphClickedPred(QCPAbstractPlottable*, int, QMouseEvent*)));
    connect(ui->Plt_PredRef_NewD, SIGNAL(plottableClick(QCPAbstractPlottable*, int, QMouseEvent*)), this, SLOT(graphClickedPredNew(QCPAbstractPlottable*, int, QMouseEvent*)));
}

MainWindow::~MainWindow()
{
    delete ui;

    //Free memory
    //delete pltspectra;
}

/*

        PARTIAL LEAST SQUARES REGRESSION ANALYSIS (PLSR)

*/

//PLSR action triggered
void MainWindow::on_actionTrain_PLSR_Model_triggered()
{
    //Empty the qlabels before opening page
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_Xrows, SLOT(clear()));
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_Xcols, SLOT(clear()));
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_Xnrows, SLOT(clear()));
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_Xncols, SLOT(clear()));
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_Yrows, SLOT(clear()));
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_Ycols, SLOT(clear()));
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_Ynrows, SLOT(clear()));
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_Yncols, SLOT(clear()));
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_SSegAns, SLOT(clear()));

    //Hide the qlabels and Spin boxes for K-Fold Cross validation
    //Qlabel
    ui->LB_NSegment->setVisible(false);
    ui->LB_SSegment->setVisible(false);
    ui->LB_SSegAns->setVisible(false);
    //Qspinbox
    ui->SP_NSegment->setVisible(false);
    //Set the initial number of segments (Default)for K-fold CV
    ui->SP_NSegment->setValue(10);

    //Empty the qlabels for R-Square and RMSE
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_RSquareTrain, SLOT(clear()));
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_RMSETrain, SLOT(clear()));

    //Setting the initial state of the radio buttons
    ui->RB_SNV->setChecked(true); //checked
    ui->RB_Leave1->setChecked(true); //checked

    //Set the initial number of components
    ui->SB_Components->setValue(30);

    //Empty the Predicted Vs Reference  Plot
    ui->Plot_PredVRef->addGraph();
    ui->Plot_PredVRef->addGraph();
    ui->Plot_PredVRef->addGraph();
    ui->Plot_PredVRef->addGraph();
    ui->Plot_PredVRef->addGraph();
    ui->Plot_PredVRef->graph(0)->data()->clear();
    ui->Plot_PredVRef->graph(1)->data()->clear();
    ui->Plot_PredVRef->graph(2)->data()->clear();
    ui->Plot_PredVRef->graph(3)->data()->clear();
    ui->Plot_PredVRef->graph(4)->data()->clear();
    ui->Plot_PredVRef->legend->clearItems();
    ui->Plot_PredVRef->legend->setVisible(false);
    ui->Plot_PredVRef->replot();

    //Empty the Beta Coefficient  Plot
    ui->Plot_Beta->addGraph();
    ui->Plot_Beta->graph(0)->data()->clear();
    ui->Plot_Beta->legend->clearItems();
    ui->Plot_Beta->replot();

    //Set text on push buttons
    ui->PB_Train->setText("Train PLSR Model");
    ui->PB_ViewPlots->setText("View PLSR Plots");
    //ui->PB_Predict->setText("Predict");

    //Set text on the Group boxes GB_ModelInput
    ui->GB_ModelInput->setTitle("PLSR Model Input");
    ui->GB_TrainModel->setTitle("Training PLSR Model");
    //ui->GB_DataPredClasfy->setTitle("DataSet to Predict");
    //ui->GB_PredClasfy->setTitle("Predict");

    //Set text on qlabels
    //ui->LB_DataPredClasfy->setText("Import DataSet to Predict:");

    //Set the initial values for Savitzky Golay derivative, polynomial and the window size
    ui->SB_SG_Derivative->setValue(2);
    ui->SB_SG_Polynomial->setValue(2);
    ui->SB_SG_Window->setValue(7);

    //Set the initial value for the moving average filter window size
    ui->SB_Smooth_Window->setValue(7);

    //Hide the qlabels and Spin boxes for Savitzky Golay preprocessing method
    //Qlabel
    ui->LB_SG_Derivative->setVisible(false);
    ui->LB_SG_Polynomial->setVisible(false);
    ui->LB_SG_Window->setVisible(false);
    //Qspinbox
    ui->SB_SG_Derivative->setVisible(false);
    ui->SB_SG_Polynomial->setVisible(false);
    ui->SB_SG_Window->setVisible(false);

    //Hide the qlabel and Spin box for Moving average filter(smoothing) preprocessing method
    //Qlabel
    ui->LB_Smooth_Window->setVisible(false);

    //Qspinbox
    ui->SB_Smooth_Window->setVisible(false);

    //Open the PLS Analysis widget
    ui->stackedWidget->setCurrentIndex(1);

}

//Import data for training from file
void MainWindow::on_PB_importText_clicked()
{
    //Create filter for text files (.txt)
    QString filter = "Text Files(*.txt)";

    //Create string for dynamic path
    QString filename = QFileDialog::getOpenFileName(this, "Load data from text files", qApp->applicationDirPath(), filter);

    //arma::mat spectra_data; Global Variable
    //arma::vec response_data; Global Variable

    Import_Data(filename.toStdString(), spectra_data, response_data);

    //Insert the number of rows and columns of input data into labels
    //Predictor X
    ui->LB_Xrows->setText(QString::number(spectra_data.n_rows));
    ui->LB_Xnrows->setText(QString::number(spectra_data.n_rows));
    ui->LB_Xcols->setText(QString::number(spectra_data.n_cols));
    ui->LB_Xncols->setText(QString::number(spectra_data.n_cols));

    //Response Y
    ui->LB_Yrows->setText(QString::number(response_data.n_rows));
    ui->LB_Ynrows->setText(QString::number(response_data.n_rows));
    ui->LB_Ycols->setText(QString::number(response_data.n_cols));
    ui->LB_Yncols->setText(QString::number(response_data.n_cols));


}

//Obtain the number of segments (K) for K-Fold CV from the user
void MainWindow::on_RB_Kfold_clicked()
{
    QMessageBox::information(this, "Number of Segments:", "Input the Number of Segments for K-Fold CV");

    //Show the qlabels and Spin boxes for K-Fold Cross validation when radio button is checked
    //qlabel
    ui->LB_NSegment->setVisible(true);
    ui->LB_SSegment->setVisible(true);
    ui->LB_SSegAns->setVisible(true);
    //Qspinbox
    ui->SP_NSegment->setVisible(true);

    //Get the Value of K from the User
    //int K_Fold = 10;
    int K_Fold_disp = ui->SP_NSegment->value();
    //Let Number of samples per segment/fold be NS
    int NS_disp = spectra_data.n_rows / K_Fold_disp;

    //Display the Number of samples per segment/fold to the User
    ui->LB_SSegAns->setText(QString::number(NS_disp));


}

//Update the Number of samples per segment
void MainWindow::on_SP_NSegment_valueChanged(int arg1)
{
    //Let Number of samples per segment/fold be NS
    int NS_disp2 = spectra_data.n_rows / arg1;

    //Display the Number of samples per segment/fold to the User
    ui->LB_SSegAns->setText(QString::number(NS_disp2));

}

//Train PLSR or PLS-DA Model
void MainWindow::on_PB_Train_clicked()
{
    mat spectra_data_prep;
    //Compute the preprocessed predictor (X) data if the radio buttons are checked
    //SNV
    if (ui->RB_SNV->isChecked())
    {
        //Perform SNV using coded function
        mat Spectra_SNV;
        Standard_Normal_variate(spectra_data, Spectra_SNV);

        spectra_data_prep = Spectra_SNV;

    }

    //Multiplicative scatter correction (MSC)
    if (ui->RB_MSC->isChecked())
    {
        //Transpose Input spectra_data so as to perform MSC with Mlpack
        mat input = trans(spectra_data);

        //Perform MSC
        mat XMSC;
        MSC_Process(input, XMSC);

        //Transpose the data to obtain the original orientation of the data after MSC preprocessing
        mat XMSC_transpose = trans(XMSC);

        //Save MSC processed data
        //XMSC_transpose.save("MSC_processed.csv", arma::csv_ascii);

        spectra_data_prep = XMSC_transpose;

    }

    //Savitzky Golay
    if (ui->RB_Savitzky->isChecked())
    {
        //Perform Savitzky-Golay smoothing/derivatization on the spectral data.
        //Transpose - carry out Savitzky-Golay smoothing/derivatization and transpose again
        mat data_t = trans(spectra_data);
        mat  SG_data_t = zeros<mat>(spectra_data.n_cols, spectra_data.n_rows);
        //polynomial_order The order of the polynomial
        //arma::uword polynomial_order = 3;
        arma::uword polynomial_order = ui->SB_SG_Polynomial->value();

        //window_size The size of the filter window.
        arma::uword window_size = ui->SB_SG_Window->value();

        //derivative_order The order of the derivative.
        arma::uword derivative_order = ui->SB_SG_Derivative->value();
        SG_data_t = sgolayfilt(data_t,polynomial_order,window_size,derivative_order,1);

        mat SG_data = trans(SG_data_t);

        //Save Savitzky-Golay data
        //SG_data.save("Savitzky_Golay_data.csv", arma::csv_ascii);

        spectra_data_prep = SG_data;

    }

    //Smoothing (Moving Average Filter)
    if (ui->RB_Smooth->isChecked())
    {
        //CreateMovingAverageFilter
        arma::uword window_size = ui->SB_Smooth_Window->value();
        vec filter = CreateMovingAverageFilter(window_size);

        // Perform moving average filtering on the spectral data.
        //Transpose - carry out smoothing and transpose again
        mat data_t = trans(spectra_data);
        mat  Smooth_data_t = zeros<mat>(spectra_data.n_cols, spectra_data.n_rows);
        for (uword j = 0; j < data_t.n_cols; ++j) {
            Smooth_data_t.col(j) = ApplyFilter(data_t.col(j), filter);
        }
        mat Smooth_data = trans(Smooth_data_t);

        //Save smoothed data
        //Smooth_data.save("Smoothed_data.csv", arma::csv_ascii);

        spectra_data_prep = Smooth_data;

    }

    //Mean normalization
    if (ui->RB_MeanNorm->isChecked())
    {
        //Perform Mean normalization
        //Transpose - carry out Mean normalization and transpose again
        mat spectra_data_t = trans(spectra_data);
        mat Mean_Normalized;
        Mean_Normalize(spectra_data_t, Mean_Normalized);

        //Save mean normalized data
        //Mean_Normalized.save("Mean_Normalized_data.csv", arma::csv_ascii);

        spectra_data_prep = Mean_Normalized;

    }

    //Min/Max Normalization
    if (ui->RB_MinMaxNorm->isChecked())
    {
        //Perform Min-Max Normalization
        //Transpose - carry out Min-Max Normalization and transpose again
        mat spectra_data_t = trans(spectra_data);
        mat Min_Max_Normalized;
        Min_Max_Normalize(spectra_data_t, Min_Max_Normalized);

        //Save mean normalized data
        //Min_Max_Normalized.save("Min_Max_Normalized_data.csv", arma::csv_ascii);

        spectra_data_prep = Min_Max_Normalized;

    }

    //Standardization
    if (ui->RB_Standardize->isChecked())
    {
        //Perform Standardization on data
        //Transpose - carry out Standardization and transpose again
        mat spectra_data_t = trans(spectra_data);
        arma::mat Standardized_data;
        Standardization(spectra_data_t, Standardized_data);

        //Save Standardized data
        //Standardized_data.save("Standardized_data.csv", arma::csv_ascii);

        spectra_data_prep = Standardized_data;

    }

    //None - No preprocessing
    if (ui->RB_NoPreproc->isChecked())
    {
        spectra_data_prep = spectra_data;

    }

    //Compute the Optimum Number of components using the checked cross validation method
    //mat Percent_Variance; //Global Variance used for plotting explained variance in y using PLS Components given by user
    int Components_Opt;   
    //Leave-one-out cross validation   
    if (ui->RB_Leave1->isChecked())
    {
        mat Xtrain = spectra_data_prep;
        mat Ytrain = response_data;

        //partial least squares regression on cross validation set
        components = ui->SB_Components->value();
        arma::mat X_loadings_CV;
        arma::mat Y_loadings_CV;
        arma::mat X_scores_CV;
        arma::mat Y_scores_CV;
        arma::mat Weights;
        arma::mat Beta_CV;
        arma::mat percent_var_CV;
        arma::mat Predicted_CV;
        double rsquared_CV;
        double RMSE_CV;

        //Mean of each column of spectral and response data (Used as a guide for mean centering the test data)
        mat XmeanTrain = arma::mean(Xtrain);
        mat YmeanTrain = arma::mean(Ytrain);

        //Create a sum of squared error matrix with zeros
        int components_p1 = components + 1;
        mat  sumsqerr = zeros<mat>(2, components_p1);

        //Fill a matrix of sum of elements of MSECV for y with zeros for each Iteration - sum is accumulating with increase in i.
        mat  MSECVy_sum = zeros<mat>(components_p1, 1);
        mat  percent_var_CV_sum = zeros<mat>(components, 1);
        //MSECVy_sum.print("MSECVy_sum Original:");

        //Create a matrix of mean of elements of MSECV for y
        mat  MSECVy_mean;
        mat  percent_var_CV_mean;

        //Leave-Out one cross validation
        for (uword r = 0; r < Xtrain.n_rows; r++)
        {
            //cout << "r1 " << r << endl;
            // extract a row of test samples for Predictor and response variables
            mat Xtest = Xtrain.row(r);
            mat Ytest = Ytrain.row(r);

            //Mean centering of the test data
            mat X0test = Xtest.each_row() - XmeanTrain;
            mat Y0test = Ytest.each_row() - YmeanTrain;

            //Remove the samples per segment (Test data removed from train data)
            Xtrain.shed_row(r);
            Ytrain.shed_row(r);
            //cout << "Number of Rows and columns in X data (Size of Matrix X): " << size(Xtrain) << endl;
            //cout << "Number of Rows and columns in Y data (Size of Matrix Y): " << size(Ytrain) << endl;

            plsregressioncv(Xtrain, Ytrain, components, X_loadings_CV, Y_loadings_CV, X_scores_CV, Y_scores_CV, Weights,
                Beta_CV, percent_var_CV, Predicted_CV, rsquared_CV, RMSE_CV);

            mat XscoreTest = X0test * Weights;

            //Sum of squared errors for the null model
            sumsqerr(0, 0) = sum(sum(arma::abs(X0test) % arma::abs(X0test), 1));
            sumsqerr(1, 0) = sum(sum(arma::abs(Y0test) % arma::abs(Y0test), 1));

            //compute sum of squared errors for models with the given components
            mat X0reconstructed;
            mat Y0reconstructed;
            for (int i = 0; i < components; ++i)
            {
                X0reconstructed = XscoreTest.cols(span(0, i)) * X_loadings_CV.cols(span(0, i)).t();
                Y0reconstructed = XscoreTest.cols(span(0, i)) * Y_loadings_CV.cols(span(0, i)).t();

                sumsqerr(0, i + 1) = sum(sum(arma::abs(X0test - X0reconstructed) % arma::abs(X0test - X0reconstructed), 1));
                sumsqerr(1, i + 1) = sum(sum(arma::abs(Y0test - Y0reconstructed) % arma::abs(Y0test - Y0reconstructed), 1));
            }

            //Trial 1
            //mat sumsqerr_reshp = reshape(sumsqerr, 1, 2*components_p1);

            //Trial 2
            //mat MSE_trial = sum(sumsqerr);
            //mat MSE_trial = sum(sumsqerr_reshp);

            uword Number_Predictions = Y0test.n_rows;
            sumsqerr /= Number_Predictions;
            //mat MSE_trial1 = reshape((MSE_trial/(Number_Predictions*k)), 2, components_p1);
            //mat MSE_trial1 = reshape((MSE_trial/Number_Predictions), 2, components_p1);

            //Transpose of Mean square errors of y (response variable)
            mat	MSE_CV_trans = sumsqerr.row(1).t();
            //mat	MSE_CV_trans = MSE_trial1.row(0).t();
            //mat	MSE_CV_trans = MSE_trial1.row(1).t();
            //MSE_CV_trans.print("MSE_CV_trans:");
            //cout << "Size of MSE_CV_trans Matrix: " << size(MSE_CV_trans) << endl;
            mat	percent_var_CV_trans = percent_var_CV.row(1).t();

            //Sum of MSE_CV
            MSECVy_sum += MSE_CV_trans;
            percent_var_CV_sum += percent_var_CV_trans;
            //MSECVy_sum.print("Sum of elements of MSECVy:");

            //Mean of MSE_CV
            MSECVy_mean = MSECVy_sum / r;
            percent_var_CV_mean = percent_var_CV_sum / r;
            //MSECVy_mean.print("MSECVy_mean:");

            ////Insert back the removed rows
            Xtrain.insert_rows(r, Xtest);
            Ytrain.insert_rows(r, Ytest);
            //cout << "Number of Rows and columns in X data (Size of Matrix X): " << size(Xtrain) << endl;
            //cout << "Number of Rows and columns in Y data (Size of Matrix Y): " << size(Ytrain) << endl;
            //cout << "r2 " << r << endl;
        }

        //MSECVy_mean.print("Final MSECVy_mean:");
        //percent_var_CV_mean.print("Final percent_var_CV_mean:");

        //Finding the optimum principal components
        //Transpose of Mean square errors of y (response variable)
        //colvec	MSE_trans = MSE.row(1).t();
        //mat	MSE_trans = MSE_cal.row(1).t();

        //Sum of all the mean square errors stored in 1x1 matrix
        //double MSE_Sum = sum(MSE_trans);
        mat MSE_Sum = sum(MSECVy_mean);

        //Sum of all the mean square errors converted to a scalar
        double MSE_Sum_sca = as_scalar(MSE_Sum);

        //Join the sum of errors to the original mean square error
        colvec	MSE_join = join_vert(MSE_Sum, MSECVy_mean(span(0, MSECVy_mean.n_rows - 2), 0));

        //Subtracting corresponding errors and dividing by the scalar sum
        mat MSE_Optimal = (MSE_join - MSECVy_mean) / MSE_Sum_sca;
        //MSE_Optimal.print("Optimal MSE :");
        //cout << "Size of Optimal MSE Matrix: " << size(MSE_Optimal) << endl;

        //Find the position corresponding to the minimum (least) MSE
        //convert MSE_trans matrix to a vector
        colvec MSE_trans_vec = vectorise(MSECVy_mean);
        uvec Min_pos = arma::find(MSE_trans_vec == min(MSE_trans_vec));

        //convert the vector to a scalar
        //uword Min_pos_sca = as_scalar(Min_pos);
        int Min_pos_sca = as_scalar(Min_pos);

        //Condition in case the minimum position is greater than the number of chosen prinicpal components
        if (Min_pos_sca > components)
        {
            Min_pos_sca -= 1;
        }

        //Condition for determining the optimimum number of pls components
        //Transpose of Percentage variance of y (response variable)
        //mat	percent_var_trans = percent_variance.row(1).t();
        //colvec	percent_var_trans = percent_var_CV.row(1).t();
        colvec	percent_var_vec = vectorise(percent_var_CV_mean);
        //percent_var_trans.print("percent_var_transpose :");
        //cout << "Size of percent_var_vec Matrix: " << size(percent_var_vec) << endl;

        //uword i = 0;
        int i = 0;

        for (; i < Min_pos_sca; i++)
            {
                if (abs(MSE_Optimal(i, 0)) <= 0.01)
                {
                    if (sum(percent_var_vec(span(0, i))) >= 0.85)
                        {
                            break;
                        }

                }

            }

        Components_Opt = i;
        Percent_Variance = percent_var_CV_mean;
        //QMessageBox::information(this, "Optimal Number of PLS Components", QString::number(Components_Opt));

    }

    //Kfold cross validation
    if (ui->RB_Kfold->isChecked())
    {
        mat Xtrain = spectra_data_prep;
        mat Ytrain = response_data;

        //partial least squares regression with K-fold Cross validation
        components = ui->SB_Components->value();
        arma::mat X_loadings_CV;
        arma::mat Y_loadings_CV;
        arma::mat X_scores_CV;
        arma::mat Y_scores_CV;
        arma::mat Weights;
        arma::mat Beta_CV;
        arma::mat percent_var_CV;
        arma::mat Predicted_CV;
        double rsquared_CV;
        double RMSE_CV;

        //Mean of each column of spectral and response data (Used as a guide for mean centering the test data)
        mat XmeanTrain = arma::mean(Xtrain);
        mat YmeanTrain = arma::mean(Ytrain);

        //Create a sum of squared error matrix with zeros
        int components_p1 = components + 1;
        mat  sumsqerr = zeros<mat>(2, components_p1);

        //K-Fold Cross Validation
        //Get the Value of K from the User
        //int K_Fold = 10;
        int K_Fold = ui->SP_NSegment->value();
        //Let Number of samples per segment/fold be NS
        int NS = Xtrain.n_rows / K_Fold;

        //Display the Number of samples per segment/fold to the User
        //ui->LB_SSegAns->setText(QString::number(NS));

        //Fill a matrix of sum of elements of MSECV for y with zeros for each Iteration - sum is accumulating with increase in i.
        mat  MSECVy_sum = zeros<mat>(components_p1, 1);
        mat  percent_var_CV_sum = zeros<mat>(components, 1);
        //MSECVy_sum.print("MSECVy_sum Original:");

        //Create a matrix of mean of elements of MSECV for y
        mat  MSECVy_mean;
        mat  percent_var_CV_mean;

        for (int k = 1; k <= K_Fold; k++)
        {
            // extract rows of samples per segment(k) for Predictor and response variables
            mat Xtest = Xtrain.rows(k*NS-NS, k*NS-1);
            mat Ytest = Ytrain.rows(k*NS-NS, k*NS-1);

            //Mean centering of the test data
            mat X0test = Xtest.each_row() - XmeanTrain;
            mat Y0test = Ytest.each_row() - YmeanTrain;

            //Remove the samples per segment (Test data removed from train data)
            Xtrain.shed_rows(k*NS-NS, k*NS-1);
            Ytrain.shed_rows(k*NS-NS, k*NS-1);
            //cout << "Number of Rows and columns in X data (Size of Matrix X): " << size(Xtrain) << endl;
            //cout << "Number of Rows and columns in Y data (Size of Matrix Y): " << size(Ytrain) << endl;

            plsregressioncv(Xtrain, Ytrain, components, X_loadings_CV, Y_loadings_CV, X_scores_CV, Y_scores_CV, Weights,
                            Beta_CV, percent_var_CV, Predicted_CV, rsquared_CV, RMSE_CV);

            mat XscoreTest = X0test * Weights;

            //Sum of squared errors for the null model
            sumsqerr(0, 0) = sum(sum(arma::abs(X0test) % arma::abs(X0test), 1));
            sumsqerr(1, 0) = sum(sum(arma::abs(Y0test) % arma::abs(Y0test), 1));

            //compute sum of squared errors for models with the given components
            mat X0reconstructed;
            mat Y0reconstructed;
            for (int i = 0; i < components; ++i)
            {
                X0reconstructed = XscoreTest.cols(span(0, i)) * X_loadings_CV.cols(span(0, i)).t();
                Y0reconstructed = XscoreTest.cols(span(0, i)) * Y_loadings_CV.cols(span(0, i)).t();

                sumsqerr(0, i + 1) = sum(sum(arma::abs(X0test - X0reconstructed) % arma::abs(X0test - X0reconstructed), 1));
                sumsqerr(1, i + 1) = sum(sum(arma::abs(Y0test - Y0reconstructed) % arma::abs(Y0test - Y0reconstructed), 1));
            }

            //Trial 1
            //mat sumsqerr_reshp = reshape(sumsqerr, 1, 2*components_p1);

            //Trial 2
            //mat MSE_trial = sum(sumsqerr);
            //mat MSE_trial = sum(sumsqerr_reshp);

            uword Number_Predictions = Y0test.n_rows;
            sumsqerr /= Number_Predictions;
            //mat MSE_trial1 = reshape((MSE_trial/(Number_Predictions*k)), 2, components_p1);
            //mat MSE_trial1 = reshape((MSE_trial/Number_Predictions), 2, components_p1);

            //Transpose of Mean square errors of y (response variable)
            mat	MSE_CV_trans = sumsqerr.row(1).t();
            //mat	MSE_CV_trans = MSE_trial1.row(0).t();
            //mat	MSE_CV_trans = MSE_trial1.row(1).t();
            //MSE_CV_trans.print("MSE_CV_trans:");
            //cout << "Size of MSE_CV_trans Matrix: " << size(MSE_CV_trans) << endl;
            mat	percent_var_CV_trans = percent_var_CV.row(1).t();

            //Sum of MSE_CV
            MSECVy_sum += MSE_CV_trans;
            percent_var_CV_sum += percent_var_CV_trans;
            //MSECVy_sum.print("Sum of elements of MSECVy:");

            //Mean of MSE_CV
            MSECVy_mean = MSECVy_sum/k;
            percent_var_CV_mean = percent_var_CV_sum / k;
            //MSECVy_mean.print("MSECVy_mean:");

            ////Insert back the removed rows
            Xtrain.insert_rows(k*NS-NS, Xtest);
            Ytrain.insert_rows(k*NS-NS, Ytest);
            //cout << "Number of Rows and columns in X data (Size of Matrix X): " << size(Xtrain) << endl;
            //cout << "Number of Rows and columns in Y data (Size of Matrix Y): " << size(Ytrain) << endl;

        }

        //MSECVy_mean.print("Final MSECVy_mean:");
        //percent_var_CV_mean.print("Final percent_var_CV_mean:");

        //Finding the optimum principal components
        //Transpose of Mean square errors of y (response variable)
        //colvec	MSE_trans = MSE.row(1).t();
        //mat	MSE_trans = MSE_cal.row(1).t();

        //Sum of all the mean square errors stored in 1x1 matrix
        //double MSE_Sum = sum(MSE_trans);
        mat MSE_Sum = sum(MSECVy_mean);

        //Sum of all the mean square errors converted to a scalar
        double MSE_Sum_sca = as_scalar(MSE_Sum);

        //Join the sum of errors to the original mean square error
        colvec	MSE_join = join_vert(MSE_Sum, MSECVy_mean(span(0, MSECVy_mean.n_rows - 2), 0));

        //Subtracting corresponding errors and dividing by the scalar sum
        mat MSE_Optimal = (MSE_join - MSECVy_mean) / MSE_Sum_sca;
        //MSE_Optimal.print("Optimal MSE :");

        //Find the position corresponding to the minimum (least) MSE
        //convert MSE_trans matrix to a vector
        colvec MSE_trans_vec = vectorise(MSECVy_mean);
        uvec Min_pos = arma::find(MSE_trans_vec == min(MSE_trans_vec));

        //convert the vector to a scalar
        //uword Min_pos_sca = as_scalar(Min_pos);
        int Min_pos_sca = as_scalar(Min_pos);

        //Condition in case the minimum position is greater than the number of chosen prinicpal components
        if (Min_pos_sca > components)
        {
            Min_pos_sca -= 1;
        }

        //Condition for determining the optimimum number of pls components
        //Transpose of Percentage variance of y (response variable)
        //mat	percent_var_trans = percent_variance.row(1).t();
        //colvec	percent_var_trans = percent_var_CV.row(1).t();
        colvec	percent_var_vec = vectorise(percent_var_CV_mean);
        //percent_var_trans.print("percent_var_transpose :");

        int i = 0;

        for (; i < Min_pos_sca; i++)
            {
                if (abs(MSE_Optimal(i, 0)) <= 0.01)
                {
                    if (sum(percent_var_vec(span(0, i))) >= 0.85)
                        {
                            break;
                        }

                }

            }

        Components_Opt = i;
        Percent_Variance = percent_var_CV_mean;
        //QMessageBox::information(this, "Optimal Number of PLS Components", QString::number(Components_Opt));


    }

    //No cross validation performed
    if (ui->RB_NoCV->isChecked())
    {
        //partial least squares regression
        //int components; //Global Variable
        components = ui->SB_Components->value();
        arma::mat X_loadings_cal;
        arma::mat Y_loadings_cal;
        arma::mat X_scores_cal;
        arma::mat Y_scores_cal;
        arma::mat Beta_cal;
        arma::mat percent_var_cal;
        arma::mat Predicted_cal;
        arma::mat MSE_cal;
        double rsquared_cal;
        double RMSE_cal;

        plsregression(spectra_data_prep, response_data, components, X_loadings_cal, Y_loadings_cal, X_scores_cal, Y_scores_cal, Beta_cal,
                        percent_var_cal, Predicted_cal, MSE_cal, rsquared_cal, RMSE_cal);

        //cout << "Size of percent_var_cal Matrix: " << arma::size(percent_var_cal);
        //cout << "Size of Predicted_cal Matrix: " << arma::size(Predicted_cal) << endl;
        //cout << "Size of MSE_cal Matrix: " << arma::size(MSE_cal) << endl;
        //MSE_cal.print("MSE: ");

        qDebug() << "rsquared_cal: " << rsquared_cal;

        //RMSE calibration
        qDebug() << "RMSE_cal: " << RMSE_cal ;

        //Finding the optimum principal components
        //Transpose of Mean square errors of y (response variable)
        //colvec	MSE_trans = MSE.row(1).t();
        //mat	MSE_trans; //Global Variable
        MSE_trans = MSE_cal.row(1).t();

        //Sum of all the mean square errors stored in 1x1 matrix
        //double MSE_Sum = sum(MSE_trans);
        mat MSE_Sum = sum(MSE_trans);

        //Sum of all the mean square errors converted to a scalar
        double MSE_Sum_sca = as_scalar(MSE_Sum);

        //Join the sum of errors to the original mean square error
        colvec	MSE_join = join_vert(MSE_Sum, MSE_trans(span(0, MSE_trans.n_rows-2), 0));

        //Subtracting corresponding errors and dividing by the scalar sum
        mat MSE_Optimal = (MSE_join - MSE_trans) / MSE_Sum_sca;
        //MSE_Optimal.print("Optimal MSE :");
        //cout << "Size of Optimal MSE Matrix: " << arma::size(MSE_Optimal) << endl;

        //Find the position corresponding to the minimum (least) MSE
        //convert MSE_trans matrix to a vector
        colvec MSE_trans_vec = vectorise(MSE_trans);
        uvec Min_pos = arma::find(MSE_trans_vec == min(MSE_trans_vec));

        //convert the vector to a scalar
        //uword Min_pos_sca = as_scalar(Min_pos);
        int Min_pos_sca = as_scalar(Min_pos);

        //Condition in case the minimum position is greater than the number of chosen prinicpal components
        if (Min_pos_sca > components)
        {
            Min_pos_sca -= 1;
        }

        //Condition for determining the optimimum number of pls components
        //Transpose of Percentage variance of y (response variable)
        //mat	percent_var_trans = percent_variance.row(1).t();
        colvec	percent_var_trans = percent_var_cal.row(1).t();
        //percent_var_trans.print("percent_var_transpose :");
        //cout << "Size of percent_var_transpose Matrix: " << arma::size(percent_var_trans) << endl;

        //uword i = 0;
        int i = 0;

        for ( ; i < Min_pos_sca; i++)
        {
            if (abs(MSE_Optimal(i, 0)) <= 0.01)
            {
                if (sum(percent_var_trans(span(0, i))) >= 0.85)
                {
                    break;
                }

            }

        }

        //i += 1;
        Components_Opt = i;
        mat	percent_var_transposey = percent_var_cal.row(1).t();
        Percent_Variance = percent_var_transposey;
        //QMessageBox::information(this, "Optimal Number of PLS Components", QString::number(Components_Opt));

    }

    //Perform Partial Least Squares Regression using the preprocessed data and optimum PLS components
    //Calculate pls parameters using the optimum pls components
    arma::mat X_loadings_optm;
    arma::mat Y_loadings_optm;
    arma::mat X_scores_optm;
    arma::mat Y_scores_optm;
    //arma::mat Beta_optm; //Global Variable
    arma::mat percent_var_optm;
    //arma::mat Predicted_optm; //Global Variable
    arma::mat MSE_optm;
    //double rsquared_optm; //Global Variable
    //double RMSE_optm; //Global Variable

    plsregression(spectra_data_prep, response_data, Components_Opt, X_loadings_optm, Y_loadings_optm, X_scores_optm, Y_scores_optm,
                    Beta_optm, percent_var_optm, Predicted_optm, MSE_optm, rsquared_optm, RMSE_optm);

    //mat Percent_Variance; //Global Variance used for plotting explained variance in x or y using optimum PLS Components
    //Percent_Variance = percent_var_optm;

    //R-squared
    //qDebug() << "rsquared_optm: " << rsquared_optm;

    //RMSE of training/calibration data set using optimum components
    //qDebug() << "RMSE_optm : " << RMSE_optm ;

    //Set the optimum PLS components to the qlabel
    ui->LB_OptimumPLSComp->setText(QString::number(Components_Opt));

    QMessageBox::information(this, "Status", "Done..!");   

}

//Open PLSR or PLSDA plots page for training data
void MainWindow::on_PB_ViewPlots_clicked()
{
    if(ui->PB_ViewPlots->text() == "View PLSR Plots")
    {
        ui->GB_PLSDA_PltDetails->hide();
        ui->GB_PLSR_PltDetails->show();
    }
    else if(ui->PB_ViewPlots->text() == "View Classification Plots")
    {
        ui->GB_PLSR_PltDetails->hide();
        ui->GB_PLSDA_PltDetails->show();

        /*Show the entries depending on the number of classes - already catered for when spinbox value changes
        if(ui->SP_Classes->value() == 2)
        {
            //Show entries for 1 and 2
            ui->LB_1stClass->setVisible(true);
            ui->SP_1stClass->setVisible(true);
            ui->LB_2ndClass->setVisible(true);
            ui->SP_2ndClass->setVisible(true);

            //hide entries for 3, 4 and 5
            ui->LB_3rdClass->setVisible(false);
            ui->SP_3rdClass->setVisible(false);
            ui->LB_4thClass->setVisible(false);
            ui->SP_4thClass->setVisible(false);
            ui->LB_5thClass->setVisible(false);
            ui->SP_5thClass->setVisible(false);

        }*/
    }

    //Open the plots widget
    ui->stackedWidget->setCurrentIndex(2);

}

//Close the application
void MainWindow::on_actionExit_triggered()
{
    QApplication::quit();
}

//Plotting for PLSR model
void MainWindow::on_PB_PlotPred_Ref_clicked()
{
    //Predicted Vs Refrence Data plot for Calibration and cross validation
    //Convert armadillo matrix or vector to c++ vectors
    //Refrence (Response)
    stdvec response_data_Cvec = conv_to< stdvec >::from(response_data);
    //Convert c++ vectors to qvector
    //int N_elements = response_data.n_rows;
    //QVector<double> response_data_qvec(N_elements);
    //response_data_qvec = QVector<double>::fromStdVector(response_data_Cvec); //fromStdVector Deprecated
    QVector<double> response_data_qvec = QVector<double>(response_data_Cvec.begin(), response_data_Cvec.end());

    //Predicted using optimum PLS components
    stdvec Predicted_optm_Cvec = conv_to< stdvec >::from(Predicted_optm);
    //Convert c++ vectors to qvector
    QVector<double> Predicted_optm_qvec = QVector<double>(Predicted_optm_Cvec.begin(), Predicted_optm_Cvec.end());

    //Set up scatter plot for Predicted Vs Refrence
    // configure right and top axis to show ticks but no labels:
    ui->Plot_PredVRef->axisRect()->setupFullAxesBox();

    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    //ui->Plot_PredVRef->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
    ui->Plot_PredVRef->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

    //Thickness of points
    const int Point_thick = 8;

    //Setup graph
    ui->Plot_PredVRef->addGraph();// add new graph
    ui->Plot_PredVRef->graph(0)->setName("Reference,Predicted");
    //ui->Plot_PredVRef->addGraph()->setName("Predicted Vs Refrence Plot for Calibration"); // add new graph
    ui->Plot_PredVRef->graph(0)->setData(response_data_qvec, Predicted_optm_qvec); // pass data points to graphs:
    ui->Plot_PredVRef->graph(0)->setPen(QPen(Qt::red)); // line color red for first graph
    ui->Plot_PredVRef->xAxis->setLabel("Reference Y");
    ui->Plot_PredVRef->yAxis->setLabel("Predicted Y");
    ui->Plot_PredVRef->graph(0)->rescaleAxes(); // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
    ui->Plot_PredVRef->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
    ui->Plot_PredVRef->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);
    ui->Plot_PredVRef->replot();

    ui->LB_RSquareTrain->setText(QString::number(rsquared_optm));
    ui->LB_RMSETrain->setText(QString::number(RMSE_optm));

}

//Show values on graph when clicked
void MainWindow::graphClickedPred(QCPAbstractPlottable * plottable, int dataIndex, QMouseEvent * event)
{
    // Predicted Vs Reference Plot
    double x = ui->Plot_PredVRef->xAxis->pixelToCoord(event->x());
    double y = ui->Plot_PredVRef->yAxis->pixelToCoord(event->y());
    //QString message = QString("%1 : (%2, %3)").arg(plottable->name()).arg(x).arg(y);
    QString message = plottable->name() + ": " + QString::number(x, 'f', 2) + " , " + QString::number(y, 'f', 2);
    QToolTip::showText(event->globalPos(), message);

}

//Beta Coeeficient Plot
void MainWindow::on_PB_Plot_Beta_clicked()
{
    //Prepare Beta Coefficient Data for Plotting
    //Remove the Intercept from the Beta Coefficient
    mat Beta_Plot = Beta_optm;
    Beta_Plot.shed_row(0);
    //Beta_Plot.print("Beta plot");

    //Convert armadillo matrix or vector to std c++ vector
    stdvec Beta_Plot_cvec = conv_to< stdvec >::from(Beta_Plot);
    //Convert c++ vectors to qvector
    QVector<double> Beta_Plot_qvec = QVector<double>(Beta_Plot_cvec.begin(), Beta_Plot_cvec.end());

    //Setup Beta Coeficient Line graph against the wavelength
    // configure right and top axis to show ticks but no labels:
    ui->Plot_Beta->axisRect()->setupFullAxesBox();

    //Thickness of lines/curves
    const int pen_width = 3;

    //Axes
    ui->Plot_Beta->xAxis->setLabel("Wavelength (nm)");
    ui->Plot_Beta->yAxis->setLabel("Beta Coefficient");

    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    ui->Plot_Beta->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

    //plot the Beta Coefficient
    ui->Plot_Beta->addGraph(); // add new graph
    ui->Plot_Beta->graph(0)->setName("Beta Coefficient"); //Set name
    ui->Plot_Beta->graph(0)->setData(Wavelength_qvec, Beta_Plot_qvec); // pass data points to graphs:
    ui->Plot_Beta->graph(0)->setPen(QPen(Qt::blue, pen_width));
    //ui->Plot_Beta->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle ));
    ui->Plot_Beta->graph(0)->setLineStyle(QCPGraph::lsLine);
    ui->Plot_Beta->graph(0)->rescaleAxes();  // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
    ui->Plot_Beta->replot();


}

//Commented out the explained Variance plot
/*
void MainWindow::on_PB_Plot_EVar_clicked()
{
    //Note: Changed graph to Explained Variance Plot
    //Measure for Prediction Error plot
    //Enables a data scientist to determine the optimal PLS components using Global and Local minimum concepts
    stdvec MSE_y_Cvec = conv_to< stdvec >::from(MSE_trans);
    //Convert c++ vectors to qvector
    QVector<double> MSE_y_qvec = QVector<double>(MSE_y_Cvec.begin(), MSE_y_Cvec.end());

    //Create a vector of numbers from 1 to number of PLS components specified by user
    //NB: many ways of implementing this - https://stackoverflow.com/questions/17694579/use-stdfill-to-populate-vector-with-increasing-numbers
    std::vector<int> components_Cvec(components+1) ; // vector with number of components plus 1 ints - MSE has plus 1.
    std::iota (std::begin(components_Cvec), std::end(components_Cvec), 1); // Fill with 1, 2,..., PLS components.
    //Convert c++ vectors to qvector
    QVector<double> components_qvec = QVector<double>(components_Cvec.begin(), components_Cvec.end());

    //Setup Line graph for calibration and cross validation
    // configure right and top axis to show ticks but no labels:
    ui->Plot_MSE->axisRect()->setupFullAxesBox();

    //Thickness of lines/curves
    const int pen_width = 3;

    //Axes
    ui->Plot_MSE->xAxis->setLabel("Number of Components");
    ui->Plot_MSE->yAxis->setLabel("Estimated Mean Squared Prediction Error");

    //Show legend
    ui->Plot_MSE->legend->setVisible(true);

    //add graph
    ui->Plot_MSE->addGraph();
    //ui->Plot_MSE->addGraph();

    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    ui->Plot_MSE->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

    //plot the mean squared prediction error
    ui->Plot_MSE->graph(0)->setName("Calibration Set"); // add new graph
    ui->Plot_MSE->graph(0)->setData(components_qvec, MSE_y_qvec); // pass data points to graphs:
    ui->Plot_MSE->graph(0)->setPen(QPen(Qt::blue, pen_width));
    ui->Plot_MSE->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc));
    ui->Plot_MSE->graph(0)->setLineStyle(QCPGraph::lsLine);
    ui->Plot_MSE->graph(0)->rescaleAxes();  //// let the ranges scale themselves so graph 0 fits perfectly in the visible area:
    ui->Plot_MSE->replot();


    //Percent Var in y
    //mat Percent_Variancey = Percent_Variance.row(1).t();
    mat Percent_Variancey = Percent_Variance;

    //cumulative sum of elements in percent_Variancey. NB -size of matrix is components by 1
    //cumsum( y ) -For matrix y, return a matrix containing the cumulative sum of elements in each column (dim=0), or each row (dim=1)

    mat percent_Variancey_cum = cumsum(Percent_Variancey, 0);
    //percent_Variancey_cum.print("percent_Variancey_cum");

    stdvec PercentVar_y_cvec = conv_to< stdvec >::from(percent_Variancey_cum);
    //Convert c++ vectors to qvector
    QVector<double> PercentVar_y_qvec = QVector<double>(PercentVar_y_cvec.begin(), PercentVar_y_cvec.end());

    //Create a vector of numbers from 1 to number of PLS components specified by user
    //NB: many ways of implementing this - https://stackoverflow.com/questions/17694579/use-stdfill-to-populate-vector-
    //with-increasing-numbers
    std::vector<int> components_Cvec(components) ; // vector with number of components.
    std::iota (std::begin(components_Cvec), std::end(components_Cvec), 1); // Fill with 1, 2,..., PLS components.
    //Convert c++ vectors to qvector
    QVector<double> components_qvec = QVector<double>(components_Cvec.begin(), components_Cvec.end());

    //Setup Explained variance Line graph for either calibration or cross validation
    // configure right and top axis to show ticks but no labels:
    ui->Plot_EV->axisRect()->setupFullAxesBox();

    //Thickness of lines/curves
    const int pen_width = 3;

    //Axes
    ui->Plot_EV->xAxis->setLabel("Number of PLS Components");
    //ui->Plot_EV->xAxis->setLabel("Optimum Number of PLS Components");
    ui->Plot_EV->yAxis->setLabel("Y-Variance");

    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    ui->Plot_EV->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

    //plot the Explained variance in y
    ui->Plot_EV->addGraph(); // add new graph
    ui->Plot_EV->graph(0)->setName("Explained Variance in y"); //Set name
    ui->Plot_EV->graph(0)->setData(components_qvec, PercentVar_y_qvec); // pass data points to graphs:
    ui->Plot_EV->graph(0)->setPen(QPen(Qt::blue, pen_width));
    ui->Plot_EV->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc));
    ui->Plot_EV->graph(0)->setLineStyle(QCPGraph::lsLine);
    ui->Plot_EV->graph(0)->rescaleAxes();  //// let the ranges scale themselves so graph 0 fits perfectly in the visible area:
    ui->Plot_EV->replot();

}*/

//actionPredict_New_Dataset
//Validate the trained PLSR Model using a test dataset
void MainWindow::on_actionPredict_New_Dataset_triggered()
{
    //Empty the qlabels before opening predict page
    connect(ui->actionPredict_New_Dataset, SIGNAL(triggered()), ui->LB_Xrows_Pred, SLOT(clear()));
    connect(ui->actionPredict_New_Dataset, SIGNAL(triggered()), ui->LB_Xcols_Pred, SLOT(clear()));
    connect(ui->actionPredict_New_Dataset, SIGNAL(triggered()), ui->LB_Xnrows_Pred, SLOT(clear()));
    connect(ui->actionPredict_New_Dataset, SIGNAL(triggered()), ui->LB_Xncols_Pred, SLOT(clear()));
    connect(ui->actionPredict_New_Dataset, SIGNAL(triggered()), ui->LB_Yrows_Pred, SLOT(clear()));
    connect(ui->actionPredict_New_Dataset, SIGNAL(triggered()), ui->LB_Ycols_Pred, SLOT(clear()));
    connect(ui->actionPredict_New_Dataset, SIGNAL(triggered()), ui->LB_Ynrows_Pred, SLOT(clear()));
    connect(ui->actionPredict_New_Dataset, SIGNAL(triggered()), ui->LB_Yncols_Pred, SLOT(clear()));

    //Empty the qlabels for R-Square and RMSE
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_RSquarePredict, SLOT(clear()));
    connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_RMSEPredict, SLOT(clear()));

    //Setting the initial state of the radio buttons
    ui->RB_SNV_Pred->setChecked(true); //checked

    //Empty the Predicted vs Reference Plot for the new Dataset
    ui->Plt_PredRef_NewD->addGraph();
    ui->Plt_PredRef_NewD->graph(0)->data()->clear();
    ui->Plt_PredRef_NewD->legend->clearItems();
    ui->Plt_PredRef_NewD->replot();

    //Set text on push buttons
    //ui->PB_Predict->setText("Predict");
    ui->PB_Predict->setText("Validate PLSR Model");
    ui->PB_ViewPlots_Pred->setText("View Validated PLSR Plots");

    //Set text on the Group boxes GB_ModelInput
    ui->GB_DataPredClasfy->setTitle("DataSet to Validate");
    ui->GB_PredClasfy->setTitle("Validate PLSR Model");

    //Set text on qlabels
    ui->LB_DataPredClasfy->setText("Import DataSet to Validate:");

    //Set the initial values for Savitzky Golay derivative, polynomial and the window size
    ui->SB_SG_Deriv_Pred->setValue(2);
    ui->SB_SG_Poly_Pred->setValue(2);
    ui->SB_SG_Wndw_Pred->setValue(7);

    //Set the initial value for the moving average filter window size
    ui->SB_Smooth_Wndw_Pred->setValue(7);

    //Hide the qlabels and Spin boxes for Savitzky Golay preprocessing method (Prediction)
    //Qlabel
    ui->LB_SG_Deriv_Pred->setVisible(false);
    ui->LB_SG_Poly_Pred->setVisible(false);
    ui->LB_SG_Wndw_Pred->setVisible(false);
    //Qspinbox
    ui->SB_SG_Deriv_Pred->setVisible(false);
    ui->SB_SG_Poly_Pred->setVisible(false);
    ui->SB_SG_Wndw_Pred->setVisible(false);

    //Hide the qlabel and Spin box for Moving average filter(smoothing) preprocessing method (Prediction)
    //Qlabel
    ui->LB_Smooth_Wndw_Pred->setVisible(false);

    //Qspinbox
    ui->SB_Smooth_Wndw_Pred->setVisible(false);

    //Open the prediction widget
    ui->stackedWidget->setCurrentIndex(3);

}

//Import New Data Set to be predicted
void MainWindow::on_PB_Data_2_Pred_clicked()
{
    //Create filter for text files (.txt) as input
    QString filter2 = "Text Files(*.txt)";

    //Create string for dynamic path
    QString filename2 = QFileDialog::getOpenFileName(this, "Load DataSet to Predict from text files", qApp->applicationDirPath(), filter2);

    //arma::mat spectra_data_Pred; Global Variable
    //arma::vec response_data_Pred; Global Variable

    Import_Data(filename2.toStdString(), spectra_data_Pred, response_data_Pred);

    //Insert the number of rows and columns of input Prediction data into labels
    //Predictor X
    ui->LB_Xrows_Pred->setText(QString::number(spectra_data_Pred.n_rows));
    ui->LB_Xnrows_Pred->setText(QString::number(spectra_data_Pred.n_rows));
    ui->LB_Xcols_Pred->setText(QString::number(spectra_data_Pred.n_cols));
    ui->LB_Xncols_Pred->setText(QString::number(spectra_data_Pred.n_cols));

    //Response Y
    ui->LB_Yrows_Pred->setText(QString::number(response_data_Pred.n_rows));
    ui->LB_Ynrows_Pred->setText(QString::number(response_data_Pred.n_rows));
    ui->LB_Ycols_Pred->setText(QString::number(response_data_Pred.n_cols));
    ui->LB_Yncols_Pred->setText(QString::number(response_data_Pred.n_cols));
}

//Predict New data set using trained PLSR Model (Beta Coefficient)
void MainWindow::on_PB_Predict_clicked()
{
    mat spectra_data_Validtn_prep;

    //Compute the preprocessed Validation X data if the radio buttons are checked    
    //SNV
    if (ui->RB_SNV_Pred->isChecked())
    {
        //Perform SNV on validation X data
        mat Spectra_SNV_Pred;
        Standard_Normal_variate(spectra_data_Pred, Spectra_SNV_Pred);

        spectra_data_Validtn_prep = Spectra_SNV_Pred;

    }

    //Multiplicative scatter correction (MSC)
    if (ui->RB_MSC_Pred->isChecked())
    {
        //Transpose Input spectra_data so as to perform MSC with Mlpack
        mat input_Pred = trans(spectra_data_Pred);

        //Perform MSC
        mat XMSC_Pred;
        MSC_Process(input_Pred, XMSC_Pred);

        //Transpose the data to obtain the original orientation of the data after MSC preprocessing
        mat XMSC_trans_Pred = trans(XMSC_Pred);

        //Save MSC processed data
        //XMSC_trans_Pred.save("MSC_processed_Pred.csv", arma::csv_ascii);

        spectra_data_Validtn_prep = XMSC_trans_Pred;

    }

    //Savitzky Golay
    if (ui->RB_Savitzky_Pred->isChecked())
    {
        //Perform Savitzky-Golay smoothing/derivatization on the spectral data.
        //Transpose - carry out Savitzky-Golay smoothing/derivatization and transpose again
        mat data_t_Pred = trans(spectra_data_Pred);
        mat  SG_data_t_Pred = zeros<mat>(spectra_data_Pred.n_cols, spectra_data_Pred.n_rows);

        //Polynomial order
        arma::uword poly_order_Pred = ui->SB_SG_Poly_Pred->value();

        //window_size The size of the filter window.
        arma::uword windw_size_Pred = ui->SB_SG_Wndw_Pred->value();

        //derivative_order The order of the derivative.
        arma::uword deriv_order_Pred = ui->SB_SG_Deriv_Pred->value();
        SG_data_t_Pred = sgolayfilt(data_t_Pred,poly_order_Pred,windw_size_Pred,deriv_order_Pred,1);

        mat SG_data_Pred = trans(SG_data_t_Pred);

        //Save Savitzky-Golay data
        //SG_data_Pred.save("Savitzky_Golay_data_Pred.csv", arma::csv_ascii);

        spectra_data_Validtn_prep = SG_data_Pred;

    }

    //Smoothing (Moving Average Filter)
    if (ui->RB_Smooth_Pred->isChecked())
    {
        //CreateMovingAverageFilter
        arma::uword wndow_size_Pred = ui->SB_Smooth_Wndw_Pred->value();
        vec filter_Pred = CreateMovingAverageFilter(wndow_size_Pred);

        // Perform moving average filtering on the spectral data.
        //Transpose - carry out smoothing and transpose again
        mat data_t_Pred = trans(spectra_data_Pred);
        mat  Smooth_data_t_P = zeros<mat>(spectra_data_Pred.n_cols, spectra_data_Pred.n_rows);
        for (uword j = 0; j < data_t_Pred.n_cols; ++j) {
            Smooth_data_t_P.col(j) = ApplyFilter(data_t_Pred.col(j), filter_Pred);
        }
        mat Smooth_data_Pred = trans(Smooth_data_t_P);

        //Save smoothed data
        //Smooth_data_Pred.save("Smoothed_data_Pred.csv", arma::csv_ascii);

        spectra_data_Validtn_prep = Smooth_data_Pred;

    }

    //Mean normalization
    if (ui->RB_MeanNorm_Pred->isChecked())
    {
        //Perform Mean normalization
        //Transpose - carry out Mean normalization and transpose again
        mat spectra_data_t_P = trans(spectra_data_Pred);
        mat Mean_Normalized_Pred;
        Mean_Normalize(spectra_data_t_P, Mean_Normalized_Pred);

        //Save mean normalized data
        //Mean_Normalized_Pred.save("Mean_Normalized_data_Pred.csv", arma::csv_ascii);

        spectra_data_Validtn_prep = Mean_Normalized_Pred;

    }

    //Min/Max Normalization
    if (ui->RB_MinMaxNorm_Pred->isChecked())
    {
        //Perform Min-Max Normalization
        //Transpose - carry out Min-Max Normalization and transpose again
        mat spectra_data_t_P = trans(spectra_data_Pred);
        mat Min_Max_Normalized_P;
        Min_Max_Normalize(spectra_data_t_P, Min_Max_Normalized_P);

        //Save mean normalized data
        //Min_Max_Normalized_P.save("Min_Max_Normalized_data_P.csv", arma::csv_ascii);

        spectra_data_Validtn_prep = Min_Max_Normalized_P;

    }

    //Standardization
    if (ui->RB_Standard_Pred->isChecked())
    {
        //Perform Standardization on data
        //Transpose - carry out Standardization and transpose again
        mat spectra_data_t_P = trans(spectra_data_Pred);
        arma::mat Standardized_data_P;
        Standardization(spectra_data_t_P, Standardized_data_P);

        //Save Standardized data
        //Standardized_data_P.save("Standardized_data_P.csv", arma::csv_ascii);

        spectra_data_Validtn_prep = Standardized_data_P;

    }

    //None - No preprocessing
    if (ui->RB_NoPreproc_Pred->isChecked())
    {
        spectra_data_Validtn_prep = spectra_data_Pred;

    }

    //join horizontally a column of ones to test data so as to match up the intercept in the beta coefficient
    mat Ones_m(spectra_data_Validtn_prep.n_rows, 1, fill::ones);
    mat spectra_test_data2 = join_horiz(Ones_m, spectra_data_Validtn_prep);

    //mat predicted_test; //Global Variable - Used in  Predicted Vs Reference Plot on New Dataset
    predicted_test = spectra_test_data2 * Beta_optm;
    //predicted_test.print("predicted_t_beta :");

    QMessageBox::information(this, "Status", "Done..!");

}

//PB_ViewPlots_Pred
//Open Plots page for validated values of test dataset
void MainWindow::on_PB_ViewPlots_Pred_clicked()
{
    //if(ui->PB_ViewPlots_Pred->text() == "View Predicted Plots")
    if(ui->PB_ViewPlots_Pred->text() == "View Validated PLSR Plots")
    {
        ui->GB_PLSDA_PltDetails_CLFY->hide();
        ui->GB_PLSR_PltDetails_New->show();
    }
    else if(ui->PB_ViewPlots_Pred->text() == "View Validated PLS-DA Plots")
    {
        ui->GB_PLSR_PltDetails_New->hide();
        ui->GB_PLSDA_PltDetails_CLFY->show();

        //Show the entries depending on the number of classes - already catered for when spinbox value changes
        /*if(ui->SP_Classes_CLFY->value() == 2)
        {
            //Show entries for 1 and 2
            ui->LB_1stClass_CLFY->setVisible(true);
            ui->SP_1stClass_CLFY->setVisible(true);
            ui->LB_2ndClass_CLFY->setVisible(true);
            ui->SP_2ndClass_CLFY->setVisible(true);

            //hide entries for 3, 4 and 5
            ui->LB_3rdClass_CLFY->setVisible(false);
            ui->SP_3rdClass_CLFY->setVisible(false);
            ui->LB_4thClass_CLFY->setVisible(false);
            ui->SP_4thClass_CLFY->setVisible(false);
            ui->LB_5thClass_CLFY->setVisible(false);
            ui->SP_5thClass_CLFY->setVisible(false);

        }*/
    }
    //Open the plots widget
    ui->stackedWidget->setCurrentIndex(4);
}

//Plot Predicted vs refrence for predicted new dataset, Also Calculate R-square and RMSE
void MainWindow::on_PB_Plt_PredRef_NewD_clicked()
{
    //Predicted Vs Refrence Data plot for New Dataset
    //Convert armadillo matrix or vector to c++ vectors
    //Reference (Response) New
    stdvec response_dataNew_Cvec = conv_to< stdvec >::from(response_data_Pred);
    //Convert c++ vectors to qvector
    QVector<double> response_dataNew_qvec = QVector<double>(response_dataNew_Cvec.begin(), response_dataNew_Cvec.end());

    //Predicted using beta coefficient from trained model
    stdvec predicted_NewD_cvec = conv_to< stdvec >::from(predicted_test);
    //Convert c++ vectors to qvector
    QVector<double> predicted_NewD_qvec = QVector<double>(predicted_NewD_cvec.begin(), predicted_NewD_cvec.end());

    //Set up scatter plot for Predicted Vs Reference of New Dataset
    // configure right and top axis to show ticks but no labels:
    ui->Plt_PredRef_NewD->axisRect()->setupFullAxesBox();

    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    //ui->Plt_PredRef_NewD->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
    ui->Plt_PredRef_NewD->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

    //Thickness of points
    const int Point_thick = 8;

    //Setup graph
    ui->Plt_PredRef_NewD->addGraph(); // add new graph
    ui->Plt_PredRef_NewD->graph(0)->setName("Reference,Predicted");
    //ui->Plt_PredRef_NewD->addGraph()->setName("Predicted Vs Refrence Plot for New Dataset"); // add new graph
    ui->Plt_PredRef_NewD->graph(0)->setData(response_dataNew_qvec, predicted_NewD_qvec); // pass data points to graphs:
    ui->Plt_PredRef_NewD->graph(0)->setPen(QPen(Qt::blue)); // line color blue for first graph
    ui->Plt_PredRef_NewD->xAxis->setLabel("Reference Y ");
    ui->Plt_PredRef_NewD->yAxis->setLabel("Predicted Y");
    ui->Plt_PredRef_NewD->graph(0)->rescaleAxes(); // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
    ui->Plt_PredRef_NewD->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc , Point_thick));
    ui->Plt_PredRef_NewD->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);
    ui->Plt_PredRef_NewD->replot();

    //Calculate and display R-square and RMSE
    //R-squared
    double TSS_test = accu((response_data_Pred - as_scalar(mean(response_data_Pred))) % (response_data_Pred - as_scalar(mean(response_data_Pred))));
    double RSS_test = accu(square(response_data_Pred - predicted_test)); // find the overall sum regardless of object type
    double rsquaredPLS_test = 1 - (RSS_test / TSS_test);
    //qDebug() << "R squared2 test: " << rsquaredPLS_test;
    ui->LB_RSquarePredict->setText(QString::number(rsquaredPLS_test));

    //RMSE of test data set
    //Root mean squared error
    //Note - I have used RSS because it has a similar expression to a section RMSE formula
    uword N_Predictions_test = response_data_Pred.n_rows;
    double RMSE_test = sqrt(RSS_test / N_Predictions_test);
    //qDebug() << "RMSE_test : " << RMSE_test;
    ui->LB_RMSEPredict->setText(QString::number(RMSE_test));

}

//Show the values when the graph points are clicked on using mouse
void MainWindow::graphClickedPredNew(QCPAbstractPlottable * plottable, int dataIndex, QMouseEvent * event)
{

    // Predicted Vs Reference Plot for New predicted data
    double x2 = ui->Plt_PredRef_NewD->xAxis->pixelToCoord(event->x());
    double y2 = ui->Plt_PredRef_NewD->yAxis->pixelToCoord(event->y());
    //QString message2 = QString("%1 : (%2, %3)").arg(plottable->name()).arg(x2).arg(y2);
    QString message2 = plottable->name() + ":" + QString::number(x2, 'f', 2) + " , " + QString::number(y2, 'f', 2);
    QToolTip::showText(event->globalPos(), message2);

}


/*

        PARTIAL LEAST SQUARES DISCRIMINANT (CLASSIFICATION) ANALYSIS (PLS-DA)

*/

//Open the PLS-DA page when the PLS-DA action is triggered
void MainWindow::on_actionTrain_PLS_DA_Model_triggered()
{
    //Empty the qlabels before opening page
    //Input Data
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Xrows, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Xcols, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Xnrows, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Xncols, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Yrows, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Ycols, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Ynrows, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Yncols, SLOT(clear()));
    //K-fold Segments
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_SSegAns, SLOT(clear()));
    //Accuracy
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Accur_1stCls, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Accur_2ndCls, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Accur_3rdCls, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Accur_4thCls, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Accur_5thCls, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Overall_Accur, SLOT(clear()));

    //Hide the qlabels and Spin boxes for K-Fold Cross validation
    //Qlabel
    ui->LB_NSegment->setVisible(false);
    ui->LB_SSegment->setVisible(false);
    ui->LB_SSegAns->setVisible(false);
    //Qspinbox
    ui->SP_NSegment->setVisible(false);
    //Set the initial number of segments (Default)for K-fold CV
    ui->SP_NSegment->setValue(10);

    //Empty the qlabels for R-Square and RMSE - Not required inn Classification
    //connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_RSquareTrain, SLOT(clear()));
    //connect(ui->actionTrain_PLSR_Model, SIGNAL(triggered()), ui->LB_RMSETrain, SLOT(clear()));

    //Setting the initial state of the radio buttons
    ui->RB_SNV->setChecked(true); //checked
    ui->RB_Leave1->setChecked(true); //checked

    //Set the initial number of components
    ui->SB_Components->setValue(30);

    //Empty the Predicted Vs Reference  Plot
    ui->Plot_PredVRef->addGraph();
    ui->Plot_PredVRef->addGraph();
    ui->Plot_PredVRef->addGraph();
    ui->Plot_PredVRef->addGraph();
    ui->Plot_PredVRef->addGraph();
    ui->Plot_PredVRef->graph(0)->data()->clear();
    ui->Plot_PredVRef->graph(1)->data()->clear();
    ui->Plot_PredVRef->graph(2)->data()->clear();
    ui->Plot_PredVRef->graph(3)->data()->clear();
    ui->Plot_PredVRef->graph(4)->data()->clear();
    ui->Plot_PredVRef->legend->clearItems();
    ui->Plot_PredVRef->legend->setVisible(false);
    ui->Plot_PredVRef->replot();

    //Empty the Beta Coefficient  Plot
    ui->Plot_Beta->addGraph();
    ui->Plot_Beta->graph(0)->data()->clear();
    ui->Plot_Beta->legend->clearItems();
    ui->Plot_Beta->replot();

    //Set text on push buttons
    ui->PB_Train->setText("Train PLS-DA Model");
    ui->PB_ViewPlots->setText("View Classification Plots");
    ui->PB_Predict->setText("Classify");

    //Set text on the Group boxes
    ui->GB_ModelInput->setTitle("PLS-DA Model Input");
    ui->GB_TrainModel->setTitle("Training PLS-DA Model");
    ui->GB_DataPredClasfy->setTitle("DataSet to Classify");
    ui->GB_PredClasfy->setTitle("Classify");

    //Set text on qlabels
    ui->LB_DataPredClasfy->setText("Import DataSet to Classify:");

    //Set the initial Details on number of classes and observations for each class
    ui->SP_Classes->setValue(2);
    ui->SP_1stClass->setValue(50);
    ui->SP_2ndClass->setValue(50);
    ui->SP_3rdClass->setValue(50);
    ui->SP_4thClass->setValue(50);
    ui->SP_5thClass->setValue(50);
    ui->SP_6thClass->setValue(50);

    //Set the maximum values for the spin boxes for the number of observations. Default max is 99
    ui->SP_1stClass->setMaximum(10000);
    ui->SP_2ndClass->setMaximum(10000);
    ui->SP_3rdClass->setMaximum(10000);
    ui->SP_4thClass->setMaximum(10000);
    ui->SP_5thClass->setMaximum(10000);
    ui->SP_6thClass->setMaximum(10000);

    //Set the initial values for Savitzky Golay derivative, polynomial and the window size
    ui->SB_SG_Derivative->setValue(2);
    ui->SB_SG_Polynomial->setValue(2);
    ui->SB_SG_Window->setValue(7);

    //Set the initial value for the moving average filter window size
    ui->SB_Smooth_Window->setValue(7);

    //Hide the qlabels and Spin boxes for Savitzky Golay preprocessing method
    //Qlabel
    ui->LB_SG_Derivative->setVisible(false);
    ui->LB_SG_Polynomial->setVisible(false);
    ui->LB_SG_Window->setVisible(false);
    //Qspinbox
    ui->SB_SG_Derivative->setVisible(false);
    ui->SB_SG_Polynomial->setVisible(false);
    ui->SB_SG_Window->setVisible(false);

    //Hide the qlabel and Spin box for Moving average filter(smoothing) preprocessing method
    //Qlabel
    ui->LB_Smooth_Window->setVisible(false);

    //Qspinbox
    ui->SB_Smooth_Window->setVisible(false);

    //Open the PLS Analysis widget
    ui->stackedWidget->setCurrentIndex(1);

}

//actionClassify_New_Dataset
//Open the page for validating the PLS-DA Model using test dataset when classify action is triggered
void MainWindow::on_actionClassify_New_Dataset_triggered()
{
    //Empty the qlabels before opening predict page
    connect(ui->actionClassify_New_Dataset, SIGNAL(triggered()), ui->LB_Xrows_Pred, SLOT(clear()));
    connect(ui->actionClassify_New_Dataset, SIGNAL(triggered()), ui->LB_Xcols_Pred, SLOT(clear()));
    connect(ui->actionClassify_New_Dataset, SIGNAL(triggered()), ui->LB_Xnrows_Pred, SLOT(clear()));
    connect(ui->actionClassify_New_Dataset, SIGNAL(triggered()), ui->LB_Xncols_Pred, SLOT(clear()));
    connect(ui->actionClassify_New_Dataset, SIGNAL(triggered()), ui->LB_Yrows_Pred, SLOT(clear()));
    connect(ui->actionClassify_New_Dataset, SIGNAL(triggered()), ui->LB_Ycols_Pred, SLOT(clear()));
    connect(ui->actionClassify_New_Dataset, SIGNAL(triggered()), ui->LB_Ynrows_Pred, SLOT(clear()));
    connect(ui->actionClassify_New_Dataset, SIGNAL(triggered()), ui->LB_Yncols_Pred, SLOT(clear()));
    //Accuracy
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Accur_1stCls_CLFY, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Accur_2ndCls_CLFY, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Accur_3rdCls_CLFY, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Accur_4thCls_CLFY, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Accur_5thCls_CLFY, SLOT(clear()));
    connect(ui->actionTrain_PLS_DA_Model, SIGNAL(triggered()), ui->LB_Overall_Accur_CLFY, SLOT(clear()));

    //Setting the initial state of the radio buttons
    ui->RB_SNV_Pred->setChecked(true); //checked

    //Empty the Predicted vs Reference Plot for the new Dataset
    ui->Plt_PredRef_NewD->addGraph();
    ui->Plt_PredRef_NewD->addGraph();
    ui->Plt_PredRef_NewD->addGraph();
    ui->Plt_PredRef_NewD->addGraph();
    ui->Plt_PredRef_NewD->addGraph();
    ui->Plt_PredRef_NewD->graph(0)->data()->clear();
    ui->Plt_PredRef_NewD->graph(1)->data()->clear();
    ui->Plt_PredRef_NewD->graph(2)->data()->clear();
    ui->Plt_PredRef_NewD->graph(3)->data()->clear();
    ui->Plt_PredRef_NewD->graph(4)->data()->clear();
    ui->Plt_PredRef_NewD->legend->clearItems();
    ui->Plt_PredRef_NewD->legend->setVisible(false);
    ui->Plt_PredRef_NewD->replot();

    //Set text on push buttons
    ui->PB_Predict->setText("Validate PLS-DA Model");
    ui->PB_ViewPlots_Pred->setText("View Validated PLS-DA Plots");

    //Set text on the Group boxes
    ui->GB_DataPredClasfy->setTitle("DataSet to Validate");
    ui->GB_PredClasfy->setTitle("Validate PLS-DA Model");

    //Set text on qlabels
    ui->LB_DataPredClasfy->setText("Import DataSet to Validate:");

    //Set the initial Details on number of classes and observations for each class
    ui->SP_Classes_CLFY->setValue(2);
    ui->SP_1stClass_CLFY->setValue(50);
    ui->SP_2ndClass_CLFY->setValue(50);
    ui->SP_3rdClass_CLFY->setValue(50);
    ui->SP_4thClass_CLFY->setValue(50);
    ui->SP_5thClass_CLFY->setValue(50);
    ui->SP_6thClass_CLFY->setValue(50);

    //Set the maximum values for the spin boxes for the number of observations. Default max is 99
    ui->SP_1stClass_CLFY->setMaximum(10000);
    ui->SP_2ndClass_CLFY->setMaximum(10000);
    ui->SP_3rdClass_CLFY->setMaximum(10000);
    ui->SP_4thClass_CLFY->setMaximum(10000);
    ui->SP_5thClass_CLFY->setMaximum(10000);
    ui->SP_6thClass_CLFY->setMaximum(10000);

    //Set the initial values for Savitzky Golay derivative, polynomial and the window size
    ui->SB_SG_Deriv_Pred->setValue(2);
    ui->SB_SG_Poly_Pred->setValue(2);
    ui->SB_SG_Wndw_Pred->setValue(7);

    //Set the initial value for the moving average filter window size
    ui->SB_Smooth_Wndw_Pred->setValue(7);

    //Hide the qlabels and Spin boxes for Savitzky Golay preprocessing method (Prediction)
    //Qlabel
    ui->LB_SG_Deriv_Pred->setVisible(false);
    ui->LB_SG_Poly_Pred->setVisible(false);
    ui->LB_SG_Wndw_Pred->setVisible(false);
    //Qspinbox
    ui->SB_SG_Deriv_Pred->setVisible(false);
    ui->SB_SG_Poly_Pred->setVisible(false);
    ui->SB_SG_Wndw_Pred->setVisible(false);

    //Hide the qlabel and Spin box for Moving average filter(smoothing) preprocessing method (Prediction)
    //Qlabel
    ui->LB_Smooth_Wndw_Pred->setVisible(false);

    //Qspinbox
    ui->SB_Smooth_Wndw_Pred->setVisible(false);

    //Open the prediction widget
    ui->stackedWidget->setCurrentIndex(3);
}

//PLS-DA training plots - what to display when the user changes the value of the spinbox for the number of classes in the dataset
void MainWindow::on_SP_Classes_valueChanged(int arg1)
{
    //Show the entries depending on the number of classes
    if(arg1 == 2)
    {
        //Show entries for 1, 2 and Overall accuracy
        //Number of Observations
        ui->LB_1stClass->setVisible(true);
        ui->SP_1stClass->setVisible(true);
        ui->LB_2ndClass->setVisible(true);
        ui->SP_2ndClass->setVisible(true);
        //Accuracy
        ui->LB_Accur_1stCls1->setVisible(true); //Label for accuracy
        ui->LB_Accur_1stCls->setVisible(true); //Value of accuracy
        ui->LB_Accur_2ndCls1->setVisible(true);
        ui->LB_Accur_2ndCls->setVisible(true);
        ui->LB_Overall_Accur1->setVisible(true);
        ui->LB_Overall_Accur->setVisible(true);


        //hide entries for 3, 4, 5, 6
        //Number of Observations
        ui->LB_3rdClass->setVisible(false);
        ui->SP_3rdClass->setVisible(false);
        ui->LB_4thClass->setVisible(false);
        ui->SP_4thClass->setVisible(false);
        ui->LB_5thClass->setVisible(false);
        ui->SP_5thClass->setVisible(false);
        ui->LB_6thClass->setVisible(false);
        ui->SP_6thClass->setVisible(false);
        //Accuracy
        ui->LB_Accur_3rdCls1->setVisible(false);
        ui->LB_Accur_3rdCls->setVisible(false);
        ui->LB_Accur_4thCls1->setVisible(false);
        ui->LB_Accur_4thCls->setVisible(false);
        ui->LB_Accur_5thCls1->setVisible(false);
        ui->LB_Accur_5thCls->setVisible(false);
        ui->LB_Accur_6thCls1->setVisible(false);
        ui->LB_Accur_6thCls->setVisible(false);

    }else if(arg1 == 3)
    {
        //Show entries for 1, 2 and 3
        //Number of Observations
        ui->LB_1stClass->setVisible(true);
        ui->SP_1stClass->setVisible(true);
        ui->LB_2ndClass->setVisible(true);
        ui->SP_2ndClass->setVisible(true);
        ui->LB_3rdClass->setVisible(true);
        ui->SP_3rdClass->setVisible(true);
        //Accuracy
        ui->LB_Accur_1stCls1->setVisible(true); //Label for accuracy
        ui->LB_Accur_1stCls->setVisible(true); //Value of accuracy
        ui->LB_Accur_2ndCls1->setVisible(true);
        ui->LB_Accur_2ndCls->setVisible(true);
        ui->LB_Accur_3rdCls1->setVisible(true);
        ui->LB_Accur_3rdCls->setVisible(true);
        ui->LB_Overall_Accur1->setVisible(true);
        ui->LB_Overall_Accur->setVisible(true);

        //hide entries for 4, 5, 6
        //Number of Observations
        ui->LB_4thClass->setVisible(false);
        ui->SP_4thClass->setVisible(false);
        ui->LB_5thClass->setVisible(false);
        ui->SP_5thClass->setVisible(false);
        ui->LB_6thClass->setVisible(false);
        ui->SP_6thClass->setVisible(false);
        //Accuracy
        ui->LB_Accur_4thCls1->setVisible(false);
        ui->LB_Accur_4thCls->setVisible(false);
        ui->LB_Accur_5thCls1->setVisible(false);
        ui->LB_Accur_5thCls->setVisible(false);
        ui->LB_Accur_6thCls1->setVisible(false);
        ui->LB_Accur_6thCls->setVisible(false);

    }else if(arg1 == 4)
    {
        //Show entries for 1, 2, 3 and 4
        //Number of Observations
        ui->LB_1stClass->setVisible(true);
        ui->SP_1stClass->setVisible(true);
        ui->LB_2ndClass->setVisible(true);
        ui->SP_2ndClass->setVisible(true);
        ui->LB_3rdClass->setVisible(true);
        ui->SP_3rdClass->setVisible(true);
        ui->LB_4thClass->setVisible(true);
        ui->SP_4thClass->setVisible(true);
        //Accuracy
        ui->LB_Accur_1stCls1->setVisible(true); //Label for accuracy
        ui->LB_Accur_1stCls->setVisible(true); //Value of accuracy
        ui->LB_Accur_2ndCls1->setVisible(true);
        ui->LB_Accur_2ndCls->setVisible(true);
        ui->LB_Accur_3rdCls1->setVisible(true);
        ui->LB_Accur_3rdCls->setVisible(true);
        ui->LB_Accur_4thCls1->setVisible(true);
        ui->LB_Accur_4thCls->setVisible(true);
        ui->LB_Overall_Accur1->setVisible(true);
        ui->LB_Overall_Accur->setVisible(true);

        //hide entries for 5, 6
        //Number of Observations
        ui->LB_5thClass->setVisible(false);
        ui->SP_5thClass->setVisible(false);
        ui->LB_6thClass->setVisible(false);
        ui->SP_6thClass->setVisible(false);
        //Accuracy
        ui->LB_Accur_5thCls1->setVisible(false);
        ui->LB_Accur_5thCls->setVisible(false);
        ui->LB_Accur_6thCls1->setVisible(false);
        ui->LB_Accur_6thCls->setVisible(false);

    }else if(arg1 == 5)
    {
        //Show entries for 1, 2, 3, 4 and 5
        //Number of Observations
        ui->LB_1stClass->setVisible(true);
        ui->SP_1stClass->setVisible(true);
        ui->LB_2ndClass->setVisible(true);
        ui->SP_2ndClass->setVisible(true);
        ui->LB_3rdClass->setVisible(true);
        ui->SP_3rdClass->setVisible(true);
        ui->LB_4thClass->setVisible(true);
        ui->SP_4thClass->setVisible(true);
        ui->LB_5thClass->setVisible(true);
        ui->SP_5thClass->setVisible(true);
        //Accuracy
        ui->LB_Accur_1stCls1->setVisible(true); //Label for accuracy
        ui->LB_Accur_1stCls->setVisible(true); //Value of accuracy
        ui->LB_Accur_2ndCls1->setVisible(true);
        ui->LB_Accur_2ndCls->setVisible(true);
        ui->LB_Accur_3rdCls1->setVisible(true);
        ui->LB_Accur_3rdCls->setVisible(true);
        ui->LB_Accur_4thCls1->setVisible(true);
        ui->LB_Accur_4thCls->setVisible(true);
        ui->LB_Accur_5thCls1->setVisible(true);
        ui->LB_Accur_5thCls->setVisible(true);
        ui->LB_Overall_Accur1->setVisible(true);
        ui->LB_Overall_Accur->setVisible(true);

        //hide entries for 6
        //Number of Observations
        ui->LB_6thClass->setVisible(false);
        ui->SP_6thClass->setVisible(false);
        //Accuracy
        ui->LB_Accur_6thCls1->setVisible(false);
        ui->LB_Accur_6thCls->setVisible(false);

    }else if(arg1 == 6)
    {
        //Show entries for 1, 2, 3, 4, 5, 6
        //Number of Observations
        ui->LB_1stClass->setVisible(true);
        ui->SP_1stClass->setVisible(true);
        ui->LB_2ndClass->setVisible(true);
        ui->SP_2ndClass->setVisible(true);
        ui->LB_3rdClass->setVisible(true);
        ui->SP_3rdClass->setVisible(true);
        ui->LB_4thClass->setVisible(true);
        ui->SP_4thClass->setVisible(true);
        ui->LB_5thClass->setVisible(true);
        ui->SP_5thClass->setVisible(true);
        ui->LB_6thClass->setVisible(true);
        ui->SP_6thClass->setVisible(true);
        //Accuracy
        ui->LB_Accur_1stCls1->setVisible(true); //Label for accuracy
        ui->LB_Accur_1stCls->setVisible(true); //Value of accuracy
        ui->LB_Accur_2ndCls1->setVisible(true);
        ui->LB_Accur_2ndCls->setVisible(true);
        ui->LB_Accur_3rdCls1->setVisible(true);
        ui->LB_Accur_3rdCls->setVisible(true);
        ui->LB_Accur_4thCls1->setVisible(true);
        ui->LB_Accur_4thCls->setVisible(true);
        ui->LB_Accur_5thCls1->setVisible(true);
        ui->LB_Accur_5thCls->setVisible(true);
        ui->LB_Accur_6thCls1->setVisible(true);
        ui->LB_Accur_6thCls->setVisible(true);
        ui->LB_Overall_Accur1->setVisible(true);
        ui->LB_Overall_Accur->setVisible(true);

        //hide entries for None

    }else{
        QMessageBox::information(this, "Warning", "Number of Classes not Currently Supported");
    }
}

//Plot for the PLS-DA Classification Model - training PLS-DA plots
void MainWindow::on_PB_PltPred_Ref_Class_clicked()
{
    //Predicted Vs Refrence Data plot for Calibration or cross validation
    //Convert armadillo matrix or vector to c++ vectors
    //Reference (Response)
    //First Class
    int F_Index_1stCls = 0;
    int L_Index_1stCls = ui->SP_1stClass->value()-1;
    vec response_1stCls = response_data(span(F_Index_1stCls, L_Index_1stCls));
    stdvec response_1stCls_Cvec = conv_to< stdvec >::from(response_1stCls);
    //Convert c++ vectors to qvector
    QVector<double> response_1stCls_qvec = QVector<double>(response_1stCls_Cvec.begin(), response_1stCls_Cvec.end());

    //Predicted using optimum PLS components
    mat Predicted_1stCls = Predicted_optm(span(0, ui->SP_1stClass->value()-1), 0);
    stdvec Predicted_1stCls_Cvec = conv_to< stdvec >::from(Predicted_1stCls);
    //Convert c++ vectors to qvector
    QVector<double> Predicted_1stCls_qvec = QVector<double>(Predicted_1stCls_Cvec.begin(), Predicted_1stCls_Cvec.end());

    //Second Class
    int F_Index_2ndCls = ui->SP_1stClass->value();
    int L_Index_2ndCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()-1;
    vec response_2ndCls = response_data(span(F_Index_2ndCls, L_Index_2ndCls));
    stdvec response_2ndCls_Cvec = conv_to< stdvec >::from(response_2ndCls);
    //Convert c++ vectors to qvector
    QVector<double> response_2ndCls_qvec = QVector<double>(response_2ndCls_Cvec.begin(), response_2ndCls_Cvec.end());

    //Predicted using optimum PLS components
    mat Predicted_2ndCls = Predicted_optm(span(F_Index_2ndCls, L_Index_2ndCls), 0);
    stdvec Predicted_2ndCls_Cvec = conv_to< stdvec >::from(Predicted_2ndCls);
    //Convert c++ vectors to qvector
    QVector<double> Predicted_2ndCls_qvec = QVector<double>(Predicted_2ndCls_Cvec.begin(), Predicted_2ndCls_Cvec.end());

    //Set up scatter plot for Predicted Vs Refrence
    // configure right and top axis to show ticks but no labels:
    ui->Plot_PredVRef->axisRect()->setupFullAxesBox();

    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    //ui->Plot_PredVRef->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
    ui->Plot_PredVRef->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

    //Thickness of points
    const int Point_thick = 8;

    //Label x and y axes
    ui->Plot_PredVRef->xAxis->setLabel("Reference Y");
    ui->Plot_PredVRef->yAxis->setLabel("Predicted Y");

    //setup legend
    ui->Plot_PredVRef->legend->setVisible(true);
    ui->Plot_PredVRef->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignTop);

    ui->Plot_PredVRef->addGraph();


    if(ui->SP_Classes->value() == 2)
    {
        //Add graph so as First class can be identified in the legend
        ui->Plot_PredVRef->addGraph();

        //Set range for x-axis
        ui->Plot_PredVRef->xAxis->setRange(-0.5, 1.5);
        ui->Plot_PredVRef->yAxis->setRange(-0.5, 1.5);

        //setup legend
        //Clear existing items first
        ui->Plot_PredVRef->legend->clearItems();
        ui->Plot_PredVRef->replot();

        //Legend for First Class
        ui->Plot_PredVRef->addGraph()->setName("First Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for first class
        ui->Plot_PredVRef->graph(0)->setData(response_1stCls_qvec, Predicted_1stCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(0)->setPen(QPen(Qt::red)); // line color red for first graph
        //ui->Plot_PredVRef->graph(0)->rescaleAxes(); // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
        ui->Plot_PredVRef->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle , Point_thick));
        ui->Plot_PredVRef->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Second Class
        ui->Plot_PredVRef->addGraph()->setName("Second Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Second class
        ui->Plot_PredVRef->graph(1)->setData(response_2ndCls_qvec, Predicted_2ndCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(1)->setPen(QPen(Qt::blue)); // line color blue for second graph
        //ui->Plot_PredVRef->graph(1)->rescaleAxes(); // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
        ui->Plot_PredVRef->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plot_PredVRef->graph(1)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        ui->Plot_PredVRef->replot();

        //Calculate Accuracy
        double baseline = 0.5;
        //First Class
        //Convert Predicted_1stCls to vector
        vec Predicted_1stCls_V = vectorise(Predicted_1stCls);
        vec  CorrectValue_1st = zeros<vec>(Predicted_1stCls.n_rows);
        for (uword i = 0; i < Predicted_1stCls.n_rows; i++)
        {
            if(Predicted_1stCls_V(i) < baseline)
            {
                CorrectValue_1st(i) = 1;
            }else
            {
                CorrectValue_1st(i) = 0;
            }
        }

        float Accuracy_1stCls = (sum(CorrectValue_1st)/Predicted_1stCls.n_rows)*100;
        ui->LB_Accur_1stCls->setText(QString::number(Accuracy_1stCls, 'f', 2)+" % ");

        //Second Class
        //Convert Predicted_2ndCls to vector
        vec Predicted_2ndCls_V = vectorise(Predicted_2ndCls);
        vec  CorrectValue_2nd = zeros<vec>(Predicted_2ndCls.n_rows);
        for (uword i = 0; i < Predicted_2ndCls.n_rows; i++)
        {
            if(Predicted_2ndCls_V(i) > baseline)
            {
                CorrectValue_2nd(i) = 1;
            }else
            {
                CorrectValue_2nd(i) = 0;
            }
        }

        float Accuracy_2ndCls = (sum(CorrectValue_2nd)/Predicted_2ndCls.n_rows)*100;
        ui->LB_Accur_2ndCls->setText(QString::number(Accuracy_2ndCls, 'f', 2)+" % ");

        //Overall Accuracy (Average of first and second class)
        float Overall_Accuracy = (Accuracy_1stCls + Accuracy_2ndCls)/2;
        ui->LB_Overall_Accur->setText(QString::number(Overall_Accuracy, 'f', 2)+" % ");


    }else if(ui->SP_Classes->value() == 3)
    {
        //Third Class
        int F_Index_3rdCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value();
        int L_Index_3rdCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value()-1;
        vec response_3rdCls = response_data(span(F_Index_3rdCls, L_Index_3rdCls));
        stdvec response_3rdCls_Cvec = conv_to< stdvec >::from(response_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> response_3rdCls_qvec = QVector<double>(response_3rdCls_Cvec.begin(), response_3rdCls_Cvec.end());

        //Predicted using optimum PLS components
        mat Predicted_3rdCls = Predicted_optm(span(F_Index_3rdCls, L_Index_3rdCls), 0);
        stdvec Predicted_3rdCls_Cvec = conv_to< stdvec >::from(Predicted_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> Predicted_3rdCls_qvec = QVector<double>(Predicted_3rdCls_Cvec.begin(), Predicted_3rdCls_Cvec.end());

        //Add graphs so as First and second classes can be identified in the legend
        ui->Plot_PredVRef->addGraph();
        ui->Plot_PredVRef->addGraph();

        //Set range for x-axis
        ui->Plot_PredVRef->xAxis->setRange(-0.5, 2.5);
        ui->Plot_PredVRef->yAxis->setRange(-0.5, 2.5);

        //setup legend
        //Clear existing items first
        ui->Plot_PredVRef->legend->clearItems();
        ui->Plot_PredVRef->replot();

        //Legend for First Class
        ui->Plot_PredVRef->addGraph()->setName("First Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for first class
        ui->Plot_PredVRef->graph(0)->setData(response_1stCls_qvec, Predicted_1stCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(0)->setPen(QPen(Qt::red)); // line color red for first graph
        //ui->Plot_PredVRef->graph(0)->rescaleAxes(); // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
        ui->Plot_PredVRef->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle , Point_thick));
        ui->Plot_PredVRef->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Second Class
        ui->Plot_PredVRef->addGraph()->setName("Second Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Second class
        ui->Plot_PredVRef->graph(1)->setData(response_2ndCls_qvec, Predicted_2ndCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(1)->setPen(QPen(Qt::blue)); // line color blue for second graph
        //ui->Plot_PredVRef->graph(1)->rescaleAxes(); // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
        ui->Plot_PredVRef->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plot_PredVRef->graph(1)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Third Class
        ui->Plot_PredVRef->addGraph()->setName("Third Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::green)); // line color green for Third graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Third class
        ui->Plot_PredVRef->graph(2)->setData(response_3rdCls_qvec, Predicted_3rdCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(2)->setPen(QPen(Qt::green)); // line color green for Third graph
        //ui->Plot_PredVRef->graph(1)->rescaleAxes(); // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
        ui->Plot_PredVRef->graph(2)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plot_PredVRef->graph(2)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        ui->Plot_PredVRef->replot();

        //Calculate Accuracy
        double baseline1 = 0.5;
        double baseline2 = 1.5;
        //First Class
        //Convert Predicted_1stCls to vector
        vec Predicted_1stCls_V = vectorise(Predicted_1stCls);
        vec  CorrectValue_1st = zeros<vec>(Predicted_1stCls.n_rows);
        for (uword i = 0; i < Predicted_1stCls.n_rows; i++)
        {
            if(Predicted_1stCls_V(i) < baseline1)
            {
                CorrectValue_1st(i) = 1;
            }else
            {
                CorrectValue_1st(i) = 0;
            }
        }

        float Accuracy_1stCls = (sum(CorrectValue_1st)/Predicted_1stCls.n_rows)*100;
        ui->LB_Accur_1stCls->setText(QString::number(Accuracy_1stCls, 'f', 2)+" % ");

        //Second Class
        //Convert Predicted_2ndCls to vector
        vec Predicted_2ndCls_V = vectorise(Predicted_2ndCls);
        vec  CorrectValue_2nd = zeros<vec>(Predicted_2ndCls.n_rows);
        for (uword i = 0; i < Predicted_2ndCls.n_rows; i++)
        {
            if(Predicted_2ndCls_V(i) > baseline1 && Predicted_2ndCls_V(i) < baseline2)
            {
                CorrectValue_2nd(i) = 1;
            }else
            {
                CorrectValue_2nd(i) = 0;
            }
        }

        float Accuracy_2ndCls = (sum(CorrectValue_2nd)/Predicted_2ndCls.n_rows)*100;
        ui->LB_Accur_2ndCls->setText(QString::number(Accuracy_2ndCls, 'f', 2)+" % ");

        //Third Class
        //Convert Predicted_3rdCls to vector
        vec Predicted_3rdCls_V = vectorise(Predicted_3rdCls);
        vec  CorrectValue_3rd = zeros<vec>(Predicted_3rdCls.n_rows);
        for (uword i = 0; i < Predicted_3rdCls.n_rows; i++)
        {
            if(Predicted_3rdCls_V(i) > baseline2)
            {
                CorrectValue_3rd(i) = 1;
            }else
            {
                CorrectValue_3rd(i) = 0;
            }
        }

        float Accuracy_3rdCls = (sum(CorrectValue_3rd)/Predicted_3rdCls.n_rows)*100;
        ui->LB_Accur_3rdCls->setText(QString::number(Accuracy_3rdCls, 'f', 2)+" % ");

        //Overall Accuracy (Average of first, second and third classes)
        float Overall_Accuracy = (Accuracy_1stCls + Accuracy_2ndCls + Accuracy_3rdCls)/3;
        ui->LB_Overall_Accur->setText(QString::number(Overall_Accuracy, 'f', 2)+" % ");


    }else if (ui->SP_Classes->value() == 4)
    {
        //Third Class
        int F_Index_3rdCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value();
        int L_Index_3rdCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value()-1;
        vec response_3rdCls = response_data(span(F_Index_3rdCls, L_Index_3rdCls));
        stdvec response_3rdCls_Cvec = conv_to< stdvec >::from(response_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> response_3rdCls_qvec = QVector<double>(response_3rdCls_Cvec.begin(), response_3rdCls_Cvec.end());

        //Predicted using optimum PLS components
        mat Predicted_3rdCls = Predicted_optm(span(F_Index_3rdCls, L_Index_3rdCls), 0);
        stdvec Predicted_3rdCls_Cvec = conv_to< stdvec >::from(Predicted_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> Predicted_3rdCls_qvec = QVector<double>(Predicted_3rdCls_Cvec.begin(), Predicted_3rdCls_Cvec.end());

        //Fourth Class
        int F_Index_4thCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value();
        int L_Index_4thCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value()+ui->SP_4thClass->value()-1;
        vec response_4thCls = response_data(span(F_Index_4thCls, L_Index_4thCls));
        stdvec response_4thCls_Cvec = conv_to< stdvec >::from(response_4thCls);
        //Convert c++ vectors to qvector
        QVector<double> response_4thCls_qvec = QVector<double>(response_4thCls_Cvec.begin(), response_4thCls_Cvec.end());

        //Predicted using optimum PLS components
        mat Predicted_4thCls = Predicted_optm(span(F_Index_4thCls, L_Index_4thCls), 0);
        stdvec Predicted_4thCls_Cvec = conv_to< stdvec >::from(Predicted_4thCls);
        //Convert c++ vectors to qvector
        QVector<double> Predicted_4thCls_qvec = QVector<double>(Predicted_4thCls_Cvec.begin(), Predicted_4thCls_Cvec.end());

        //Add graphs so as First, second and third classes can be identified in the legend
        ui->Plot_PredVRef->addGraph();
        ui->Plot_PredVRef->addGraph();
        ui->Plot_PredVRef->addGraph();

        //Set range for x-axis
        ui->Plot_PredVRef->xAxis->setRange(-0.5, 3.5);
        ui->Plot_PredVRef->yAxis->setRange(-0.5, 3.5);

        //setup legend
        //Clear existing items first
        ui->Plot_PredVRef->legend->clearItems();
        ui->Plot_PredVRef->replot();

        //Legend for First Class
        ui->Plot_PredVRef->addGraph()->setName("First Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for first class
        ui->Plot_PredVRef->graph(0)->setData(response_1stCls_qvec, Predicted_1stCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(0)->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plot_PredVRef->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle , Point_thick));
        ui->Plot_PredVRef->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Second Class
        ui->Plot_PredVRef->addGraph()->setName("Second Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Second class
        ui->Plot_PredVRef->graph(1)->setData(response_2ndCls_qvec, Predicted_2ndCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(1)->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plot_PredVRef->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plot_PredVRef->graph(1)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Third Class
        ui->Plot_PredVRef->addGraph()->setName("Third Class"); // add new graph
        //ui->Plot_PredVRef->graph()->setPen(QPen(Qt::green)); // line color green for Third graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::darkGreen)); // line color green for Third graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Third class
        ui->Plot_PredVRef->graph(2)->setData(response_3rdCls_qvec, Predicted_3rdCls_qvec); // pass data points to graphs:
        //ui->Plot_PredVRef->graph(2)->setPen(QPen(Qt::green)); // line color green for Third graph
        ui->Plot_PredVRef->graph(2)->setPen(QPen(Qt::darkGreen)); // line color green for Third graph
        ui->Plot_PredVRef->graph(2)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plot_PredVRef->graph(2)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Fourth Class
        ui->Plot_PredVRef->addGraph()->setName("Fourth Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Fourth class
        ui->Plot_PredVRef->graph(3)->setData(response_4thCls_qvec, Predicted_4thCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(3)->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plot_PredVRef->graph(3)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plot_PredVRef->graph(3)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        ui->Plot_PredVRef->replot();

        //Calculate Accuracy
        double baseline1 = 0.5;
        double baseline2 = 1.5;
        double baseline3 = 2.5;
        //First Class
        //Convert Predicted_1stCls to vector
        vec Predicted_1stCls_V = vectorise(Predicted_1stCls);
        vec  CorrectValue_1st = zeros<vec>(Predicted_1stCls.n_rows);
        for (uword i = 0; i < Predicted_1stCls.n_rows; i++)
        {
            if(Predicted_1stCls_V(i) < baseline1)
            {
                CorrectValue_1st(i) = 1;
            }else
            {
                CorrectValue_1st(i) = 0;
            }
        }

        float Accuracy_1stCls = (sum(CorrectValue_1st)/Predicted_1stCls.n_rows)*100;
        ui->LB_Accur_1stCls->setText(QString::number(Accuracy_1stCls, 'f', 2)+" % ");

        //Second Class
        //Convert Predicted_2ndCls to vector
        vec Predicted_2ndCls_V = vectorise(Predicted_2ndCls);
        vec  CorrectValue_2nd = zeros<vec>(Predicted_2ndCls.n_rows);
        for (uword i = 0; i < Predicted_2ndCls.n_rows; i++)
        {
            if(Predicted_2ndCls_V(i) > baseline1 && Predicted_2ndCls_V(i) < baseline2)
            {
                CorrectValue_2nd(i) = 1;
            }else
            {
                CorrectValue_2nd(i) = 0;
            }
        }

        float Accuracy_2ndCls = (sum(CorrectValue_2nd)/Predicted_2ndCls.n_rows)*100;
        ui->LB_Accur_2ndCls->setText(QString::number(Accuracy_2ndCls, 'f', 2)+" % ");

        //Third Class
        //Convert Predicted_3rdCls to vector
        vec Predicted_3rdCls_V = vectorise(Predicted_3rdCls);
        vec  CorrectValue_3rd = zeros<vec>(Predicted_3rdCls.n_rows);
        for (uword i = 0; i < Predicted_3rdCls.n_rows; i++)
        {
            if(Predicted_3rdCls_V(i) > baseline2 && Predicted_3rdCls_V(i) < baseline3)
            {
                CorrectValue_3rd(i) = 1;
            }else
            {
                CorrectValue_3rd(i) = 0;
            }
        }

        float Accuracy_3rdCls = (sum(CorrectValue_3rd)/Predicted_3rdCls.n_rows)*100;
        ui->LB_Accur_3rdCls->setText(QString::number(Accuracy_3rdCls, 'f', 2)+" % ");

        //Fourth Class
        //Convert Predicted_3rdCls to vector
        vec Predicted_4thCls_V = vectorise(Predicted_4thCls);
        vec  CorrectValue_4th = zeros<vec>(Predicted_4thCls.n_rows);
        for (uword i = 0; i < Predicted_4thCls.n_rows; i++)
        {
            if(Predicted_4thCls(i) > baseline3)
            {
                CorrectValue_4th(i) = 1;
            }else
            {
                CorrectValue_4th(i) = 0;
            }
        }

        float Accuracy_4thCls = (sum(CorrectValue_4th)/Predicted_4thCls.n_rows)*100;
        ui->LB_Accur_4thCls->setText(QString::number(Accuracy_4thCls, 'f', 2)+" % ");

        //Overall Accuracy (Average of first, second, third and fourth classes)
        float Overall_Accuracy = (Accuracy_1stCls + Accuracy_2ndCls + Accuracy_3rdCls + Accuracy_4thCls)/4;
        ui->LB_Overall_Accur->setText(QString::number(Overall_Accuracy, 'f', 2)+" % ");

    }else if (ui->SP_Classes->value() == 5)
    {
        //Third Class
        int F_Index_3rdCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value();
        int L_Index_3rdCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value()-1;
        vec response_3rdCls = response_data(span(F_Index_3rdCls, L_Index_3rdCls));
        stdvec response_3rdCls_Cvec = conv_to< stdvec >::from(response_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> response_3rdCls_qvec = QVector<double>(response_3rdCls_Cvec.begin(), response_3rdCls_Cvec.end());

        //Predicted using optimum PLS components
        mat Predicted_3rdCls = Predicted_optm(span(F_Index_3rdCls, L_Index_3rdCls), 0);
        stdvec Predicted_3rdCls_Cvec = conv_to< stdvec >::from(Predicted_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> Predicted_3rdCls_qvec = QVector<double>(Predicted_3rdCls_Cvec.begin(), Predicted_3rdCls_Cvec.end());

        //Fourth Class
        int F_Index_4thCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value();
        int L_Index_4thCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value()+ui->SP_4thClass->value()-1;
        vec response_4thCls = response_data(span(F_Index_4thCls, L_Index_4thCls));
        stdvec response_4thCls_Cvec = conv_to< stdvec >::from(response_4thCls);
        //Convert c++ vectors to qvector
        QVector<double> response_4thCls_qvec = QVector<double>(response_4thCls_Cvec.begin(), response_4thCls_Cvec.end());

        //Predicted using optimum PLS components
        mat Predicted_4thCls = Predicted_optm(span(F_Index_4thCls, L_Index_4thCls), 0);
        stdvec Predicted_4thCls_Cvec = conv_to< stdvec >::from(Predicted_4thCls);
        //Convert c++ vectors to qvector
        QVector<double> Predicted_4thCls_qvec = QVector<double>(Predicted_4thCls_Cvec.begin(), Predicted_4thCls_Cvec.end());

        //Fifth Class
        int F_Index_5thCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value()+ui->SP_4thClass->value();
        int L_Index_5thCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value()+ui->SP_4thClass->value()+ui->SP_5thClass->value()-1;
        vec response_5thCls = response_data(span(F_Index_5thCls, L_Index_5thCls));
        stdvec response_5thCls_Cvec = conv_to< stdvec >::from(response_5thCls);
        //Convert c++ vectors to qvector
        QVector<double> response_5thCls_qvec = QVector<double>(response_5thCls_Cvec.begin(), response_5thCls_Cvec.end());

        //Predicted using optimum PLS components
        mat Predicted_5thCls = Predicted_optm(span(F_Index_5thCls, L_Index_5thCls), 0);
        stdvec Predicted_5thCls_Cvec = conv_to< stdvec >::from(Predicted_5thCls);
        //Convert c++ vectors to qvector
        QVector<double> Predicted_5thCls_qvec = QVector<double>(Predicted_5thCls_Cvec.begin(), Predicted_5thCls_Cvec.end());

        //OR - Fifth Class implemented differently (tail)
        /*int Last_Elements = ui->SP_5thClass->value();
        vec response_5thCls = response_data.tail(Last_Elements);
        stdvec response_5thCls_Cvec = conv_to< stdvec >::from(response_5thCls);
        //Convert c++ vectors to qvector
        QVector<double> response_5thCls_qvec = QVector<double>(response_5thCls_Cvec.begin(), response_5thCls_Cvec.end());

        //Predicted using optimum PLS components
        mat Predicted_5thCls = Predicted_optm.tail_rows(Last_Elements);
        stdvec Predicted_5thCls_Cvec = conv_to< stdvec >::from(Predicted_5thCls);
        //Convert c++ vectors to qvector
        QVector<double> Predicted_5thCls_qvec = QVector<double>(Predicted_5thCls_Cvec.begin(), Predicted_5thCls_Cvec.end());*/

        //Add graphs so as First, second, third and Fourth classes can be identified in the legend
        ui->Plot_PredVRef->addGraph();
        ui->Plot_PredVRef->addGraph();
        ui->Plot_PredVRef->addGraph();
        ui->Plot_PredVRef->addGraph();

        //Set range for x-axis
        ui->Plot_PredVRef->xAxis->setRange(-0.5, 4.5);
        ui->Plot_PredVRef->yAxis->setRange(-0.5, 4.5);

        //setup legend
        //Clear existing items first
        ui->Plot_PredVRef->legend->clearItems();
        ui->Plot_PredVRef->replot();

        //Legend for First Class
        ui->Plot_PredVRef->addGraph()->setName("First Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for first class
        ui->Plot_PredVRef->graph(0)->setData(response_1stCls_qvec, Predicted_1stCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(0)->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plot_PredVRef->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle , Point_thick));
        ui->Plot_PredVRef->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Second Class
        ui->Plot_PredVRef->addGraph()->setName("Second Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Second class
        ui->Plot_PredVRef->graph(1)->setData(response_2ndCls_qvec, Predicted_2ndCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(1)->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plot_PredVRef->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plot_PredVRef->graph(1)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Third Class
        ui->Plot_PredVRef->addGraph()->setName("Third Class"); // add new graph
        //ui->Plot_PredVRef->graph()->setPen(QPen(Qt::green)); // line color green for Third graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::darkGreen)); // line color green for Third graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Third class
        ui->Plot_PredVRef->graph(2)->setData(response_3rdCls_qvec, Predicted_3rdCls_qvec); // pass data points to graphs:
        //ui->Plot_PredVRef->graph(2)->setPen(QPen(Qt::green)); // line color green for Third graph
        ui->Plot_PredVRef->graph(2)->setPen(QPen(Qt::darkGreen)); // line color green for Third graph
        ui->Plot_PredVRef->graph(2)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plot_PredVRef->graph(2)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Fourth Class
        ui->Plot_PredVRef->addGraph()->setName("Fourth Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Fourth class
        ui->Plot_PredVRef->graph(3)->setData(response_4thCls_qvec, Predicted_4thCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(3)->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plot_PredVRef->graph(3)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plot_PredVRef->graph(3)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Fifth Class
        ui->Plot_PredVRef->addGraph()->setName("Fifth Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::darkYellow)); // line color darkYellow for Fifth graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Fifth class
        ui->Plot_PredVRef->graph(4)->setData(response_5thCls_qvec, Predicted_5thCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(4)->setPen(QPen(Qt::darkYellow)); // line color darkYellow for Fifth graph
        ui->Plot_PredVRef->graph(4)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle, Point_thick));
        ui->Plot_PredVRef->graph(4)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        ui->Plot_PredVRef->replot();

        //Calculate Accuracy
        double baseline1 = 0.5;
        double baseline2 = 1.5;
        double baseline3 = 2.5;
        double baseline4 = 3.5;
        //First Class
        //Convert Predicted_1stCls to vector
        vec Predicted_1stCls_V = vectorise(Predicted_1stCls);
        vec  CorrectValue_1st = zeros<vec>(Predicted_1stCls.n_rows);
        for (uword i = 0; i < Predicted_1stCls.n_rows; i++)
        {
            if(Predicted_1stCls_V(i) < baseline1)
            {
                CorrectValue_1st(i) = 1;
            }else
            {
                CorrectValue_1st(i) = 0;
            }
        }

        float Accuracy_1stCls = (sum(CorrectValue_1st)/Predicted_1stCls.n_rows)*100;
        ui->LB_Accur_1stCls->setText(QString::number(Accuracy_1stCls, 'f', 2)+" % ");

        //Second Class
        //Convert Predicted_2ndCls to vector
        vec Predicted_2ndCls_V = vectorise(Predicted_2ndCls);
        vec  CorrectValue_2nd = zeros<vec>(Predicted_2ndCls.n_rows);
        for (uword i = 0; i < Predicted_2ndCls.n_rows; i++)
        {
            if(Predicted_2ndCls_V(i) > baseline1 && Predicted_2ndCls_V(i) < baseline2)
            {
                CorrectValue_2nd(i) = 1;
            }else
            {
                CorrectValue_2nd(i) = 0;
            }
        }

        float Accuracy_2ndCls = (sum(CorrectValue_2nd)/Predicted_2ndCls.n_rows)*100;
        ui->LB_Accur_2ndCls->setText(QString::number(Accuracy_2ndCls, 'f', 2)+" % ");

        //Third Class
        //Convert Predicted_3rdCls to vector
        vec Predicted_3rdCls_V = vectorise(Predicted_3rdCls);
        vec  CorrectValue_3rd = zeros<vec>(Predicted_3rdCls.n_rows);
        for (uword i = 0; i < Predicted_3rdCls.n_rows; i++)
        {
            if(Predicted_3rdCls_V(i) > baseline2 && Predicted_3rdCls_V(i) < baseline3)
            {
                CorrectValue_3rd(i) = 1;
            }else
            {
                CorrectValue_3rd(i) = 0;
            }
        }

        float Accuracy_3rdCls = (sum(CorrectValue_3rd)/Predicted_3rdCls.n_rows)*100;
        ui->LB_Accur_3rdCls->setText(QString::number(Accuracy_3rdCls, 'f', 2)+" % ");

        //Fourth Class
        //Convert Predicted_4thCls to vector
        vec Predicted_4thCls_V = vectorise(Predicted_4thCls);
        vec  CorrectValue_4th = zeros<vec>(Predicted_4thCls.n_rows);
        for (uword i = 0; i < Predicted_4thCls.n_rows; i++)
        {
            if(Predicted_4thCls(i) > baseline3 && Predicted_4thCls(i) < baseline4)
            {
                CorrectValue_4th(i) = 1;
            }else
            {
                CorrectValue_4th(i) = 0;
            }
        }

        float Accuracy_4thCls = (sum(CorrectValue_4th)/Predicted_4thCls.n_rows)*100;
        ui->LB_Accur_4thCls->setText(QString::number(Accuracy_4thCls, 'f', 2)+" % ");

        //Fifth Class
        //Convert Predicted_5thCls to vector
        vec Predicted_5thCls_V = vectorise(Predicted_5thCls);
        vec  CorrectValue_5th = zeros<vec>(Predicted_5thCls.n_rows);
        for (uword i = 0; i < Predicted_5thCls.n_rows; i++)
        {
            if(Predicted_5thCls(i) > baseline4)
            {
                CorrectValue_5th(i) = 1;
            }else
            {
                CorrectValue_5th(i) = 0;
            }
        }

        float Accuracy_5thCls = (sum(CorrectValue_5th)/Predicted_5thCls.n_rows)*100;
        ui->LB_Accur_5thCls->setText(QString::number(Accuracy_5thCls, 'f', 2)+" % ");

        //Overall Accuracy (Average of first, second, third, fourth and fifth classes)
        float Overall_Accuracy = (Accuracy_1stCls + Accuracy_2ndCls + Accuracy_3rdCls + Accuracy_4thCls + Accuracy_5thCls)/5;
        ui->LB_Overall_Accur->setText(QString::number(Overall_Accuracy, 'f', 2)+" % ");

    }else if (ui->SP_Classes->value() == 6)
    {
        //Third Class
        int F_Index_3rdCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value();
        int L_Index_3rdCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value()-1;
        vec response_3rdCls = response_data(span(F_Index_3rdCls, L_Index_3rdCls));
        stdvec response_3rdCls_Cvec = conv_to< stdvec >::from(response_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> response_3rdCls_qvec = QVector<double>(response_3rdCls_Cvec.begin(), response_3rdCls_Cvec.end());

        //Predicted using optimum PLS components
        mat Predicted_3rdCls = Predicted_optm(span(F_Index_3rdCls, L_Index_3rdCls), 0);
        stdvec Predicted_3rdCls_Cvec = conv_to< stdvec >::from(Predicted_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> Predicted_3rdCls_qvec = QVector<double>(Predicted_3rdCls_Cvec.begin(), Predicted_3rdCls_Cvec.end());

        //Fourth Class
        int F_Index_4thCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value();
        int L_Index_4thCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value()+ui->SP_4thClass->value()-1;
        vec response_4thCls = response_data(span(F_Index_4thCls, L_Index_4thCls));
        stdvec response_4thCls_Cvec = conv_to< stdvec >::from(response_4thCls);
        //Convert c++ vectors to qvector
        QVector<double> response_4thCls_qvec = QVector<double>(response_4thCls_Cvec.begin(), response_4thCls_Cvec.end());

        //Predicted using optimum PLS components
        mat Predicted_4thCls = Predicted_optm(span(F_Index_4thCls, L_Index_4thCls), 0);
        stdvec Predicted_4thCls_Cvec = conv_to< stdvec >::from(Predicted_4thCls);
        //Convert c++ vectors to qvector
        QVector<double> Predicted_4thCls_qvec = QVector<double>(Predicted_4thCls_Cvec.begin(), Predicted_4thCls_Cvec.end());

        //Fifth Class
        int F_Index_5thCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value()+ui->SP_4thClass->value();
        int L_Index_5thCls = ui->SP_1stClass->value()+ui->SP_2ndClass->value()+ui->SP_3rdClass->value()+ui->SP_4thClass->value()+ui->SP_5thClass->value()-1;
        vec response_5thCls = response_data(span(F_Index_5thCls, L_Index_5thCls));
        stdvec response_5thCls_Cvec = conv_to< stdvec >::from(response_5thCls);
        //Convert c++ vectors to qvector
        QVector<double> response_5thCls_qvec = QVector<double>(response_5thCls_Cvec.begin(), response_5thCls_Cvec.end());

        //Predicted using optimum PLS components
        mat Predicted_5thCls = Predicted_optm(span(F_Index_5thCls, L_Index_5thCls), 0);
        stdvec Predicted_5thCls_Cvec = conv_to< stdvec >::from(Predicted_5thCls);
        //Convert c++ vectors to qvector
        QVector<double> Predicted_5thCls_qvec = QVector<double>(Predicted_5thCls_Cvec.begin(), Predicted_5thCls_Cvec.end());

        //Sixth Class implemented differently (tail)
        int Last_Elements = ui->SP_6thClass->value();
        vec response_6thCls = response_data.tail(Last_Elements);
        stdvec response_6thCls_Cvec = conv_to< stdvec >::from(response_6thCls);
        //Convert c++ vectors to qvector
        QVector<double> response_6thCls_qvec = QVector<double>(response_6thCls_Cvec.begin(), response_6thCls_Cvec.end());

        //Predicted using optimum PLS components
        mat Predicted_6thCls = Predicted_optm.tail_rows(Last_Elements);
        stdvec Predicted_6thCls_Cvec = conv_to< stdvec >::from(Predicted_6thCls);
        //Convert c++ vectors to qvector
        QVector<double> Predicted_6thCls_qvec = QVector<double>(Predicted_6thCls_Cvec.begin(), Predicted_6thCls_Cvec.end());

        //Add graphs so as First, second, third, Fourth and fifth classes can be identified in the legend
        ui->Plot_PredVRef->addGraph();
        ui->Plot_PredVRef->addGraph();
        ui->Plot_PredVRef->addGraph();
        ui->Plot_PredVRef->addGraph();
        ui->Plot_PredVRef->addGraph();

        //Set range for x-axis
        ui->Plot_PredVRef->xAxis->setRange(-0.5, 5.5);
        ui->Plot_PredVRef->yAxis->setRange(-0.5, 5.5);

        //setup legend
        //Clear existing items first
        ui->Plot_PredVRef->legend->clearItems();
        ui->Plot_PredVRef->replot();

        //Legend for First Class
        ui->Plot_PredVRef->addGraph()->setName("First Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for first class
        ui->Plot_PredVRef->graph(0)->setData(response_1stCls_qvec, Predicted_1stCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(0)->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plot_PredVRef->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle , Point_thick));
        ui->Plot_PredVRef->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Second Class
        ui->Plot_PredVRef->addGraph()->setName("Second Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Second class
        ui->Plot_PredVRef->graph(1)->setData(response_2ndCls_qvec, Predicted_2ndCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(1)->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plot_PredVRef->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plot_PredVRef->graph(1)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Third Class
        ui->Plot_PredVRef->addGraph()->setName("Third Class"); // add new graph
        //ui->Plot_PredVRef->graph()->setPen(QPen(Qt::green)); // line color green for Third graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::darkGreen)); // line color green for Third graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Third class
        ui->Plot_PredVRef->graph(2)->setData(response_3rdCls_qvec, Predicted_3rdCls_qvec); // pass data points to graphs:
        //ui->Plot_PredVRef->graph(2)->setPen(QPen(Qt::green)); // line color green for Third graph
        ui->Plot_PredVRef->graph(2)->setPen(QPen(Qt::darkGreen)); // line color green for Third graph
        ui->Plot_PredVRef->graph(2)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plot_PredVRef->graph(2)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Fourth Class
        ui->Plot_PredVRef->addGraph()->setName("Fourth Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Fourth class
        ui->Plot_PredVRef->graph(3)->setData(response_4thCls_qvec, Predicted_4thCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(3)->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plot_PredVRef->graph(3)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plot_PredVRef->graph(3)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Fifth Class
        ui->Plot_PredVRef->addGraph()->setName("Fifth Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::darkYellow)); // line color darkYellow for Fifth graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Fifth class
        ui->Plot_PredVRef->graph(4)->setData(response_5thCls_qvec, Predicted_5thCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(4)->setPen(QPen(Qt::darkYellow)); // line color darkYellow for Fifth graph
        ui->Plot_PredVRef->graph(4)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle, Point_thick));
        ui->Plot_PredVRef->graph(4)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Sixth Class
        ui->Plot_PredVRef->addGraph()->setName("Sixth Class"); // add new graph
        ui->Plot_PredVRef->graph()->setPen(QPen(Qt::black)); // line color black for sixth graph
        ui->Plot_PredVRef->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDiamond, Point_thick));
        ui->Plot_PredVRef->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Sixth class
        ui->Plot_PredVRef->graph(5)->setData(response_6thCls_qvec, Predicted_6thCls_qvec); // pass data points to graphs:
        ui->Plot_PredVRef->graph(5)->setPen(QPen(Qt::black)); // line color black for sixth graph
        ui->Plot_PredVRef->graph(5)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDiamond, Point_thick));
        ui->Plot_PredVRef->graph(5)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        ui->Plot_PredVRef->replot();

        //Calculate Accuracy
        double baseline1 = 0.5;
        double baseline2 = 1.5;
        double baseline3 = 2.5;
        double baseline4 = 3.5;
        double baseline5 = 4.5;
        //First Class
        //First Class
        //Convert Predicted_1stCls to vector
        vec Predicted_1stCls_V = vectorise(Predicted_1stCls);
        vec  CorrectValue_1st = zeros<vec>(Predicted_1stCls.n_rows);
        for (uword i = 0; i < Predicted_1stCls.n_rows; i++)
        {
            if(Predicted_1stCls_V(i) < baseline1)
            {
                CorrectValue_1st(i) = 1;
            }else
            {
                CorrectValue_1st(i) = 0;
            }
        }

        float Accuracy_1stCls = (sum(CorrectValue_1st)/Predicted_1stCls.n_rows)*100;
        ui->LB_Accur_1stCls->setText(QString::number(Accuracy_1stCls, 'f', 2)+" % ");

        //Second Class
        //Convert Predicted_2ndCls to vector
        vec Predicted_2ndCls_V = vectorise(Predicted_2ndCls);
        vec  CorrectValue_2nd = zeros<vec>(Predicted_2ndCls.n_rows);
        for (uword i = 0; i < Predicted_2ndCls.n_rows; i++)
        {
            if(Predicted_2ndCls_V(i) > baseline1 && Predicted_2ndCls_V(i) < baseline2)
            {
                CorrectValue_2nd(i) = 1;
            }else
            {
                CorrectValue_2nd(i) = 0;
            }
        }

        float Accuracy_2ndCls = (sum(CorrectValue_2nd)/Predicted_2ndCls.n_rows)*100;
        ui->LB_Accur_2ndCls->setText(QString::number(Accuracy_2ndCls, 'f', 2)+" % ");

        //Third Class
        //Convert Predicted_3rdCls to vector
        vec Predicted_3rdCls_V = vectorise(Predicted_3rdCls);
        vec  CorrectValue_3rd = zeros<vec>(Predicted_3rdCls.n_rows);
        for (uword i = 0; i < Predicted_3rdCls.n_rows; i++)
        {
            if(Predicted_3rdCls_V(i) > baseline2 && Predicted_3rdCls_V(i) < baseline3)
            {
                CorrectValue_3rd(i) = 1;
            }else
            {
                CorrectValue_3rd(i) = 0;
            }
        }

        float Accuracy_3rdCls = (sum(CorrectValue_3rd)/Predicted_3rdCls.n_rows)*100;
        ui->LB_Accur_3rdCls->setText(QString::number(Accuracy_3rdCls, 'f', 2)+" % ");

        //Fourth Class
        //Convert Predicted_4thCls to vector
        vec Predicted_4thCls_V = vectorise(Predicted_4thCls);
        vec  CorrectValue_4th = zeros<vec>(Predicted_4thCls.n_rows);
        for (uword i = 0; i < Predicted_4thCls.n_rows; i++)
        {
            if(Predicted_4thCls(i) > baseline3 && Predicted_4thCls(i) < baseline4)
            {
                CorrectValue_4th(i) = 1;
            }else
            {
                CorrectValue_4th(i) = 0;
            }
        }

        float Accuracy_4thCls = (sum(CorrectValue_4th)/Predicted_4thCls.n_rows)*100;
        ui->LB_Accur_4thCls->setText(QString::number(Accuracy_4thCls, 'f', 2)+" % ");

        //Fifth Class
        //Convert Predicted_5thCls to vector
        vec Predicted_5thCls_V = vectorise(Predicted_5thCls);
        vec  CorrectValue_5th = zeros<vec>(Predicted_5thCls.n_rows);
        for (uword i = 0; i < Predicted_5thCls.n_rows; i++)
        {
            if(Predicted_5thCls(i) > baseline4 && Predicted_5thCls(i) < baseline5)
            {
                CorrectValue_5th(i) = 1;
            }else
            {
                CorrectValue_5th(i) = 0;
            }
        }

        float Accuracy_5thCls = (sum(CorrectValue_5th)/Predicted_5thCls.n_rows)*100;
        ui->LB_Accur_5thCls->setText(QString::number(Accuracy_5thCls, 'f', 2)+" % ");

        //Sixth Class
        //Convert Predicted_6thCls to vector
        vec Predicted_6thCls_V = vectorise(Predicted_6thCls);
        vec  CorrectValue_6th = zeros<vec>(Predicted_6thCls.n_rows);
        for (uword i = 0; i < Predicted_6thCls.n_rows; i++)
        {
            if(Predicted_6thCls(i) > baseline5)
            {
                CorrectValue_6th(i) = 1;
            }else
            {
                CorrectValue_6th(i) = 0;
            }
        }

        float Accuracy_6thCls = (sum(CorrectValue_6th)/Predicted_6thCls.n_rows)*100;
        ui->LB_Accur_6thCls->setText(QString::number(Accuracy_6thCls, 'f', 2)+" % ");

        //Overall Accuracy (Average of first, second, third, fourth, fifth and sixth classes)
        float Overall_Accuracy = (Accuracy_1stCls + Accuracy_2ndCls + Accuracy_3rdCls + Accuracy_4thCls + Accuracy_5thCls + Accuracy_5thCls)/6;
        ui->LB_Overall_Accur->setText(QString::number(Overall_Accuracy, 'f', 2)+" % ");

    }else{
        QMessageBox::information(this, "Warning", "Number of Classes not Currently Supported");
    }


}


//CLASSIFY plots on New dataset - what to display when the user changes the value of the spinbox for the number of classes in the new dataset
void MainWindow::on_SP_Classes_CLFY_valueChanged(int arg1)
{
    //Show the entries depending on the number of classes
    if(arg1 == 2)
    {
        //Show entries for 1, 2 and Overall accuracy
        //Number of Observations
        ui->LB_1stClass_CLFY->setVisible(true);
        ui->SP_1stClass_CLFY->setVisible(true);
        ui->LB_2ndClass_CLFY->setVisible(true);
        ui->SP_2ndClass_CLFY->setVisible(true);
        //Accuracy
        ui->LB_Accur_1stCls1_CLFY->setVisible(true); //Label for accuracy
        ui->LB_Accur_1stCls_CLFY->setVisible(true); //Value of accuracy
        ui->LB_Accur_2ndCls1_CLFY->setVisible(true);
        ui->LB_Accur_2ndCls_CLFY->setVisible(true);
        ui->LB_Overall_Accur1_CLFY->setVisible(true);
        ui->LB_Overall_Accur_CLFY->setVisible(true);


        //hide entries for 3, 4, 5, 6
        //Number of Observations
        ui->LB_3rdClass_CLFY->setVisible(false);
        ui->SP_3rdClass_CLFY->setVisible(false);
        ui->LB_4thClass_CLFY->setVisible(false);
        ui->SP_4thClass_CLFY->setVisible(false);
        ui->LB_5thClass_CLFY->setVisible(false);
        ui->SP_5thClass_CLFY->setVisible(false);
        ui->LB_6thClass_CLFY->setVisible(false);
        ui->SP_6thClass_CLFY->setVisible(false);
        //Accuracy
        ui->LB_Accur_3rdCls1_CLFY->setVisible(false);
        ui->LB_Accur_3rdCls_CLFY->setVisible(false);
        ui->LB_Accur_4thCls1_CLFY->setVisible(false);
        ui->LB_Accur_4thCls_CLFY->setVisible(false);
        ui->LB_Accur_5thCls1_CLFY->setVisible(false);
        ui->LB_Accur_5thCls_CLFY->setVisible(false);
        ui->LB_Accur_6thCls1_CLFY->setVisible(false);
        ui->LB_Accur_6thCls_CLFY->setVisible(false);

    }else if(arg1 == 3)
    {
        //Show entries for 1, 2 and 3
        //Number of Observations
        ui->LB_1stClass_CLFY->setVisible(true);
        ui->SP_1stClass_CLFY->setVisible(true);
        ui->LB_2ndClass_CLFY->setVisible(true);
        ui->SP_2ndClass_CLFY->setVisible(true);
        ui->LB_3rdClass_CLFY->setVisible(true);
        ui->SP_3rdClass_CLFY->setVisible(true);
        //Accuracy
        ui->LB_Accur_1stCls1_CLFY->setVisible(true); //Label for accuracy
        ui->LB_Accur_1stCls_CLFY->setVisible(true); //Value of accuracy
        ui->LB_Accur_2ndCls1_CLFY->setVisible(true);
        ui->LB_Accur_2ndCls_CLFY->setVisible(true);
        ui->LB_Accur_3rdCls1_CLFY->setVisible(true);
        ui->LB_Accur_3rdCls_CLFY->setVisible(true);
        ui->LB_Overall_Accur1_CLFY->setVisible(true);
        ui->LB_Overall_Accur_CLFY->setVisible(true);

        //hide entries for 4, 5, 6
        //Number of Observations
        ui->LB_4thClass_CLFY->setVisible(false);
        ui->SP_4thClass_CLFY->setVisible(false);
        ui->LB_5thClass_CLFY->setVisible(false);
        ui->SP_5thClass_CLFY->setVisible(false);
        ui->LB_6thClass_CLFY->setVisible(false);
        ui->SP_6thClass_CLFY->setVisible(false);
        //Accuracy
        ui->LB_Accur_4thCls1_CLFY->setVisible(false);
        ui->LB_Accur_4thCls_CLFY->setVisible(false);
        ui->LB_Accur_5thCls1_CLFY->setVisible(false);
        ui->LB_Accur_5thCls_CLFY->setVisible(false);
        ui->LB_Accur_6thCls1_CLFY->setVisible(false);
        ui->LB_Accur_6thCls_CLFY->setVisible(false);

    }else if(arg1 == 4)
    {
        //Show entries for 1, 2, 3 and 4
        //Number of Observations
        ui->LB_1stClass_CLFY->setVisible(true);
        ui->SP_1stClass_CLFY->setVisible(true);
        ui->LB_2ndClass_CLFY->setVisible(true);
        ui->SP_2ndClass_CLFY->setVisible(true);
        ui->LB_3rdClass_CLFY->setVisible(true);
        ui->SP_3rdClass_CLFY->setVisible(true);
        ui->LB_4thClass_CLFY->setVisible(true);
        ui->SP_4thClass_CLFY->setVisible(true);
        //Accuracy
        ui->LB_Accur_1stCls1_CLFY->setVisible(true); //Label for accuracy
        ui->LB_Accur_1stCls_CLFY->setVisible(true); //Value of accuracy
        ui->LB_Accur_2ndCls1_CLFY->setVisible(true);
        ui->LB_Accur_2ndCls_CLFY->setVisible(true);
        ui->LB_Accur_3rdCls1_CLFY->setVisible(true);
        ui->LB_Accur_3rdCls_CLFY->setVisible(true);
        ui->LB_Accur_4thCls1_CLFY->setVisible(true);
        ui->LB_Accur_4thCls_CLFY->setVisible(true);
        ui->LB_Overall_Accur1_CLFY->setVisible(true);
        ui->LB_Overall_Accur_CLFY->setVisible(true);

        //hide entries for 5, 6
        //Number of Observations
        ui->LB_5thClass_CLFY->setVisible(false);
        ui->SP_5thClass_CLFY->setVisible(false);
        ui->LB_6thClass_CLFY->setVisible(false);
        ui->SP_6thClass_CLFY->setVisible(false);
        //Accuracy
        ui->LB_Accur_5thCls1_CLFY->setVisible(false);
        ui->LB_Accur_5thCls_CLFY->setVisible(false);
        ui->LB_Accur_6thCls1_CLFY->setVisible(false);
        ui->LB_Accur_6thCls_CLFY->setVisible(false);

    }else if(arg1 == 5)
    {
        //Show entries for 1, 2, 3, 4 and 5
        //Number of Observations
        ui->LB_1stClass_CLFY->setVisible(true);
        ui->SP_1stClass_CLFY->setVisible(true);
        ui->LB_2ndClass_CLFY->setVisible(true);
        ui->SP_2ndClass_CLFY->setVisible(true);
        ui->LB_3rdClass_CLFY->setVisible(true);
        ui->SP_3rdClass_CLFY->setVisible(true);
        ui->LB_4thClass_CLFY->setVisible(true);
        ui->SP_4thClass_CLFY->setVisible(true);
        ui->LB_5thClass_CLFY->setVisible(true);
        ui->SP_5thClass_CLFY->setVisible(true);
        //Accuracy
        ui->LB_Accur_1stCls1_CLFY->setVisible(true); //Label for accuracy
        ui->LB_Accur_1stCls_CLFY->setVisible(true); //Value of accuracy
        ui->LB_Accur_2ndCls1_CLFY->setVisible(true);
        ui->LB_Accur_2ndCls_CLFY->setVisible(true);
        ui->LB_Accur_3rdCls1_CLFY->setVisible(true);
        ui->LB_Accur_3rdCls_CLFY->setVisible(true);
        ui->LB_Accur_4thCls1_CLFY->setVisible(true);
        ui->LB_Accur_4thCls_CLFY->setVisible(true);
        ui->LB_Accur_5thCls1_CLFY->setVisible(true);
        ui->LB_Accur_5thCls_CLFY->setVisible(true);
        ui->LB_Overall_Accur1_CLFY->setVisible(true);
        ui->LB_Overall_Accur1_CLFY->setVisible(true);

        //hide entries for 6
        //Number of Observations
        ui->LB_6thClass_CLFY->setVisible(false);
        ui->SP_6thClass_CLFY->setVisible(false);
        //Accuracy
        ui->LB_Accur_6thCls1_CLFY->setVisible(false);
        ui->LB_Accur_6thCls_CLFY->setVisible(false);

    }else if(arg1 == 6)
    {
        //Show entries for 1, 2, 3, 4, 5, 6
        //Number of Observations
        ui->LB_1stClass_CLFY->setVisible(true);
        ui->SP_1stClass_CLFY->setVisible(true);
        ui->LB_2ndClass_CLFY->setVisible(true);
        ui->SP_2ndClass_CLFY->setVisible(true);
        ui->LB_3rdClass_CLFY->setVisible(true);
        ui->SP_3rdClass_CLFY->setVisible(true);
        ui->LB_4thClass_CLFY->setVisible(true);
        ui->SP_4thClass_CLFY->setVisible(true);
        ui->LB_5thClass_CLFY->setVisible(true);
        ui->SP_5thClass_CLFY->setVisible(true);
        ui->LB_6thClass_CLFY->setVisible(true);
        ui->SP_6thClass_CLFY->setVisible(true);
        //Accuracy
        ui->LB_Accur_1stCls1_CLFY->setVisible(true); //Label for accuracy
        ui->LB_Accur_1stCls_CLFY->setVisible(true); //Value of accuracy
        ui->LB_Accur_2ndCls1_CLFY->setVisible(true);
        ui->LB_Accur_2ndCls_CLFY->setVisible(true);
        ui->LB_Accur_3rdCls1_CLFY->setVisible(true);
        ui->LB_Accur_3rdCls_CLFY->setVisible(true);
        ui->LB_Accur_4thCls1_CLFY->setVisible(true);
        ui->LB_Accur_4thCls_CLFY->setVisible(true);
        ui->LB_Accur_5thCls1_CLFY->setVisible(true);
        ui->LB_Accur_5thCls_CLFY->setVisible(true);
        ui->LB_Accur_6thCls1_CLFY->setVisible(true);
        ui->LB_Accur_6thCls_CLFY->setVisible(true);
        ui->LB_Overall_Accur1_CLFY->setVisible(true);
        ui->LB_Overall_Accur1_CLFY->setVisible(true);

        //hide entries for none

    }else{
        QMessageBox::information(this, "Warning", "Number of Classes not Currently Supported");
    }
}

//Plot for the classified New dataset - testing on new dataset
void MainWindow::on_PB_PltPred_Ref_Class_CLFY_clicked()
{
    //Predicted Vs Refrence Data plot for Classified new dataset
    //Convert armadillo matrix or vector to c++ vectors
    //Reference (Response)
    //First Class
    int F_Index_1stCls_CLF = 0;
    int L_Index_1stCls_CLF = ui->SP_1stClass_CLFY->value()-1;
    vec response_1stCls_CLFY = response_data_Pred(span(F_Index_1stCls_CLF, L_Index_1stCls_CLF));
    stdvec response_1stCls_CLFY_Cvec = conv_to< stdvec >::from(response_1stCls_CLFY);
    //Convert c++ vectors to qvector
    QVector<double> response_1stCls_CLFY_qvec = QVector<double>(response_1stCls_CLFY_Cvec.begin(), response_1stCls_CLFY_Cvec.end());

    //Classified using beta coefficient from trained model
    mat Classified_1stCls = predicted_test(span(0, L_Index_1stCls_CLF), 0);
    stdvec Classified_1stCls_Cvec = conv_to< stdvec >::from(Classified_1stCls);
    //Convert c++ vectors to qvector
    QVector<double> Classified_1stCls_qvec = QVector<double>(Classified_1stCls_Cvec.begin(), Classified_1stCls_Cvec.end());

    //Second Class
    int F_Index_2ndCls_CLF = ui->SP_1stClass_CLFY->value();
    int L_Index_2ndCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()-1;
    vec response_2ndCls_CLFY = response_data_Pred(span(F_Index_2ndCls_CLF, L_Index_2ndCls_CLF));
    stdvec response_2ndCls_CLFY_Cvec = conv_to< stdvec >::from(response_2ndCls_CLFY);
    //Convert c++ vectors to qvector
    QVector<double> response_2ndCls_CLFY_qvec = QVector<double>(response_2ndCls_CLFY_Cvec.begin(), response_2ndCls_CLFY_Cvec.end());

    //Predicted using beta coefficient from trained model
    mat Classified_2ndCls = predicted_test(span(F_Index_2ndCls_CLF, L_Index_2ndCls_CLF), 0);
    stdvec Classified_2ndCls_Cvec = conv_to< stdvec >::from(Classified_2ndCls);
    //Convert c++ vectors to qvector
    QVector<double> Classified_2ndCls_qvec = QVector<double>(Classified_2ndCls_Cvec.begin(), Classified_2ndCls_Cvec.end());

    //Set up scatter plot for Predicted Vs Reference for new classified dataset
    // configure right and top axis to show ticks but no labels:
    ui->Plt_PredRef_NewD->axisRect()->setupFullAxesBox();

    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    //ui->Plot_PredVRef->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
    ui->Plt_PredRef_NewD->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

    //Thickness of points
    const int Point_thick = 8;

    //Label x and y axes
    ui->Plt_PredRef_NewD->xAxis->setLabel("Reference Y");
    ui->Plt_PredRef_NewD->yAxis->setLabel("Predicted Y");

    //setup legend
    ui->Plt_PredRef_NewD->legend->setVisible(true);
    ui->Plt_PredRef_NewD->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignTop);

    ui->Plt_PredRef_NewD->addGraph();


    if(ui->SP_Classes_CLFY->value() == 2)
    {
        //Add graph so as First class can be identified in the legend
        ui->Plt_PredRef_NewD->addGraph();

        //Set range for x-axis
        ui->Plt_PredRef_NewD->xAxis->setRange(-0.5, 1.5);
        ui->Plt_PredRef_NewD->yAxis->setRange(-0.5, 1.5);

        //setup legend
        //Clear existing items first
        ui->Plt_PredRef_NewD->legend->clearItems();
        ui->Plt_PredRef_NewD->replot();

        //Legend for First Class
        ui->Plt_PredRef_NewD->addGraph()->setName("First Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for first class
        ui->Plt_PredRef_NewD->graph(0)->setData(response_1stCls_CLFY_qvec, Classified_1stCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(0)->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plt_PredRef_NewD->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle , Point_thick));
        ui->Plt_PredRef_NewD->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Second Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Second Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Second class
        ui->Plt_PredRef_NewD->graph(1)->setData(response_2ndCls_CLFY_qvec, Classified_2ndCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(1)->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plt_PredRef_NewD->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plt_PredRef_NewD->graph(1)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        ui->Plt_PredRef_NewD->replot();

        //Calculate Accuracy
        double baseline = 0.5;
        //First Class
        //Convert Classified_1stCls to vector
        vec Classified_1stCls_V = vectorise(Classified_1stCls);
        vec  CorrectValue_1st_CLF = zeros<vec>(Classified_1stCls.n_rows);
        for (uword i = 0; i < Classified_1stCls.n_rows; i++)
        {
            if(Classified_1stCls_V(i) < baseline)
            {
                CorrectValue_1st_CLF(i) = 1;
            }else
            {
                CorrectValue_1st_CLF(i) = 0;
            }
        }

        float Accuracy_1stCls_CLF = (sum(CorrectValue_1st_CLF)/Classified_1stCls.n_rows)*100;
        ui->LB_Accur_1stCls_CLFY->setText(QString::number(Accuracy_1stCls_CLF, 'f', 2)+" % ");

        //Second Class
        //Convert Classified_2ndCls to vector
        vec Classified_2ndCls_V = vectorise(Classified_2ndCls);
        vec  CorrectValue_2nd_CLF = zeros<vec>(Classified_2ndCls.n_rows);
        for (uword i = 0; i < Classified_2ndCls.n_rows; i++)
        {
            if(Classified_2ndCls_V(i) > baseline)
            {
                CorrectValue_2nd_CLF(i) = 1;
            }else
            {
                CorrectValue_2nd_CLF(i) = 0;
            }
        }

        float Accuracy_2ndCls_CLF = (sum(CorrectValue_2nd_CLF)/Classified_2ndCls.n_rows)*100;
        ui->LB_Accur_2ndCls_CLFY->setText(QString::number(Accuracy_2ndCls_CLF, 'f', 2)+" % ");

        //Overall Accuracy (Average of first and second class)
        float Overall_Accuracy_CLF = (Accuracy_1stCls_CLF + Accuracy_2ndCls_CLF)/2;
        ui->LB_Overall_Accur_CLFY->setText(QString::number(Overall_Accuracy_CLF, 'f', 2)+" % ");


    }else if(ui->SP_Classes_CLFY->value() == 3)
    {
        //Third Class
        int F_Index_3rdCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value();
        int L_Index_3rdCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value()-1;
        vec response_3rdCls_CLF = response_data_Pred(span(F_Index_3rdCls_CLF, L_Index_3rdCls_CLF));
        stdvec response_3rdCls_CLF_Cvec = conv_to< stdvec >::from(response_3rdCls_CLF);
        //Convert c++ vectors to qvector
        QVector<double> response_3rdCls_CLF_qvec = QVector<double>(response_3rdCls_CLF_Cvec.begin(), response_3rdCls_CLF_Cvec.end());

        //Classified using Beta Coefficient
        mat Classified_3rdCls = predicted_test(span(F_Index_3rdCls_CLF, L_Index_3rdCls_CLF), 0);
        stdvec Classified_3rdCls_Cvec = conv_to< stdvec >::from(Classified_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> Classified_3rdCls_qvec = QVector<double>(Classified_3rdCls_Cvec.begin(), Classified_3rdCls_Cvec.end());

        //Add graphs so as First and second classes can be identified in the legend
        ui->Plt_PredRef_NewD->addGraph();
        ui->Plt_PredRef_NewD->addGraph();

        //Set range for x-axis
        ui->Plt_PredRef_NewD->xAxis->setRange(-0.5, 2.5);
        ui->Plt_PredRef_NewD->yAxis->setRange(-0.5, 2.5);

        //setup legend
        //Clear existing items first
        ui->Plt_PredRef_NewD->legend->clearItems();
        ui->Plt_PredRef_NewD->replot();

        //Legend for First Class
        ui->Plt_PredRef_NewD->addGraph()->setName("First Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for first class
        ui->Plt_PredRef_NewD->graph(0)->setData(response_1stCls_CLFY_qvec, Classified_1stCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(0)->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plt_PredRef_NewD->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle , Point_thick));
        ui->Plt_PredRef_NewD->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Second Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Second Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Second class
        ui->Plt_PredRef_NewD->graph(1)->setData(response_2ndCls_CLFY_qvec, Classified_2ndCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(1)->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plt_PredRef_NewD->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plt_PredRef_NewD->graph(1)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Third Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Third Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::green)); // line color green for Third graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Third class
        ui->Plt_PredRef_NewD->graph(2)->setData(response_3rdCls_CLF_qvec, Classified_3rdCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(2)->setPen(QPen(Qt::green)); // line color green for Third graph
        ui->Plt_PredRef_NewD->graph(2)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plt_PredRef_NewD->graph(2)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        ui->Plt_PredRef_NewD->replot();

        //Calculate Accuracy
        double baseline1 = 0.5;
        double baseline2 = 1.5;
        //First Class
        //Convert Classified_1stCls to vector
        vec Classified_1stCls_V = vectorise(Classified_1stCls);
        vec  CorrectValue_1st_CLF = zeros<vec>(Classified_1stCls.n_rows);
        for (uword i = 0; i < Classified_1stCls.n_rows; i++)
        {
            if(Classified_1stCls_V(i) < baseline1)
            {
                CorrectValue_1st_CLF(i) = 1;
            }else
            {
                CorrectValue_1st_CLF(i) = 0;
            }
        }

        float Accuracy_1stCls_CLF = (sum(CorrectValue_1st_CLF)/Classified_1stCls.n_rows)*100;
        ui->LB_Accur_1stCls_CLFY->setText(QString::number(Accuracy_1stCls_CLF, 'f', 2)+" % ");

        //Second Class
        //Convert Classified_2ndCls to vector
        vec Classified_2ndCls_V = vectorise(Classified_2ndCls);
        vec  CorrectValue_2nd_CLF = zeros<vec>(Classified_2ndCls.n_rows);
        for (uword i = 0; i < Classified_2ndCls.n_rows; i++)
        {
            if(Classified_2ndCls_V(i) > baseline1 && Classified_2ndCls_V(i) < baseline2)
            {
                CorrectValue_2nd_CLF(i) = 1;
            }else
            {
                CorrectValue_2nd_CLF(i) = 0;
            }
        }

        float Accuracy_2ndCls_CLF = (sum(CorrectValue_2nd_CLF)/Classified_2ndCls.n_rows)*100;
        ui->LB_Accur_2ndCls_CLFY->setText(QString::number(Accuracy_2ndCls_CLF, 'f', 2)+" % ");

        //Third Class
        //Convert Classified_3rdCls to vector
        vec Classified_3rdCls_V = vectorise(Classified_3rdCls);
        vec  CorrectValue_3rd_CLF = zeros<vec>(Classified_3rdCls.n_rows);
        for (uword i = 0; i < Classified_3rdCls.n_rows; i++)
        {
            if(Classified_3rdCls_V(i) > baseline2)
            {
                CorrectValue_3rd_CLF(i) = 1;
            }else
            {
                CorrectValue_3rd_CLF(i) = 0;
            }
        }

        float Accuracy_3rdCls_CLF = (sum(CorrectValue_3rd_CLF)/Classified_3rdCls.n_rows)*100;
        ui->LB_Accur_3rdCls_CLFY->setText(QString::number(Accuracy_3rdCls_CLF, 'f', 2)+" % ");

        //Overall Accuracy (Average of first, second and third classes)
        float Overall_Accuracy_CLF = (Accuracy_1stCls_CLF + Accuracy_2ndCls_CLF + Accuracy_3rdCls_CLF)/3;
        ui->LB_Overall_Accur_CLFY->setText(QString::number(Overall_Accuracy_CLF, 'f', 2)+" % ");


    }else if (ui->SP_Classes_CLFY->value() == 4)
    {
        //Third Class
        int F_Index_3rdCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value();
        int L_Index_3rdCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value()-1;
        vec response_3rdCls_CLF = response_data_Pred(span(F_Index_3rdCls_CLF, L_Index_3rdCls_CLF));
        stdvec response_3rdCls_CLF_Cvec = conv_to< stdvec >::from(response_3rdCls_CLF);
        //Convert c++ vectors to qvector
        QVector<double> response_3rdCls_CLF_qvec = QVector<double>(response_3rdCls_CLF_Cvec.begin(), response_3rdCls_CLF_Cvec.end());

        //Classified using Beta Coefficient
        mat Classified_3rdCls = predicted_test(span(F_Index_3rdCls_CLF, L_Index_3rdCls_CLF), 0);
        stdvec Classified_3rdCls_Cvec = conv_to< stdvec >::from(Classified_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> Classified_3rdCls_qvec = QVector<double>(Classified_3rdCls_Cvec.begin(), Classified_3rdCls_Cvec.end());

        //Fourth Class
        int F_Index_4thCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value();
        int L_Index_4thCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value()+ui->SP_4thClass_CLFY->value()-1;
        vec response_4thCls_CLF = response_data_Pred(span(F_Index_4thCls_CLF, L_Index_4thCls_CLF));
        stdvec response_4thCls_CLF_Cvec = conv_to< stdvec >::from(response_4thCls_CLF);
        //Convert c++ vectors to qvector
        QVector<double> response_4thCls_CLF_qvec = QVector<double>(response_4thCls_CLF_Cvec.begin(), response_4thCls_CLF_Cvec.end());

        //Classified using Beta Coefficient
        mat Classified_4thCls = predicted_test(span(F_Index_4thCls_CLF, L_Index_4thCls_CLF), 0);
        stdvec Classified_4thCls_Cvec = conv_to< stdvec >::from(Classified_4thCls);
        //Convert c++ vectors to qvector
        QVector<double> Classified_4thCls_qvec = QVector<double>(Classified_4thCls_Cvec.begin(), Classified_4thCls_Cvec.end());

        //Add graphs so as First, second and third classes can be identified in the legend
        ui->Plt_PredRef_NewD->addGraph();
        ui->Plt_PredRef_NewD->addGraph();
        ui->Plt_PredRef_NewD->addGraph();

        //Set range for x-axis
        ui->Plt_PredRef_NewD->xAxis->setRange(-0.5, 3.5);
        ui->Plt_PredRef_NewD->yAxis->setRange(-0.5, 3.5);

        //setup legend
        //Clear existing items first
        ui->Plt_PredRef_NewD->legend->clearItems();
        ui->Plt_PredRef_NewD->replot();

        //Legend for First Class
        ui->Plt_PredRef_NewD->addGraph()->setName("First Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for first class
        ui->Plt_PredRef_NewD->graph(0)->setData(response_1stCls_CLFY_qvec, Classified_1stCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(0)->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plt_PredRef_NewD->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle , Point_thick));
        ui->Plt_PredRef_NewD->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Second Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Second Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Second class
        ui->Plt_PredRef_NewD->graph(1)->setData(response_2ndCls_CLFY_qvec, Classified_2ndCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(1)->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plt_PredRef_NewD->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plt_PredRef_NewD->graph(1)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Third Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Third Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::darkGreen)); // line color green for Third graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Third class
        ui->Plt_PredRef_NewD->graph(2)->setData(response_3rdCls_CLF_qvec, Classified_3rdCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(2)->setPen(QPen(Qt::darkGreen)); // line color darkGreen for Third graph
        ui->Plt_PredRef_NewD->graph(2)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plt_PredRef_NewD->graph(2)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Fourth Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Fourth Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Fourth class
        ui->Plt_PredRef_NewD->graph(3)->setData(response_4thCls_CLF_qvec, Classified_4thCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(3)->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plt_PredRef_NewD->graph(3)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plt_PredRef_NewD->graph(3)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        ui->Plt_PredRef_NewD->replot();

        //Calculate Accuracy
        double baseline1 = 0.5;
        double baseline2 = 1.5;
        double baseline3 = 2.5;
        //First Class
        //Convert Classified_1stCls to vector
        vec Classified_1stCls_V = vectorise(Classified_1stCls);
        vec  CorrectValue_1st_CLF = zeros<vec>(Classified_1stCls.n_rows);
        for (uword i = 0; i < Classified_1stCls.n_rows; i++)
        {
            if(Classified_1stCls_V(i) < baseline1)
            {
                CorrectValue_1st_CLF(i) = 1;
            }else
            {
                CorrectValue_1st_CLF(i) = 0;
            }
        }

        float Accuracy_1stCls_CLF = (sum(CorrectValue_1st_CLF)/Classified_1stCls.n_rows)*100;
        ui->LB_Accur_1stCls_CLFY->setText(QString::number(Accuracy_1stCls_CLF, 'f', 2)+" % ");

        //Second Class
        //Convert Predicted_2ndCls to vector
        vec Classified_2ndCls_V = vectorise(Classified_2ndCls);
        vec  CorrectValue_2nd_CLF = zeros<vec>(Classified_2ndCls.n_rows);
        for (uword i = 0; i < Classified_2ndCls.n_rows; i++)
        {
            if(Classified_2ndCls_V(i) > baseline1 && Classified_2ndCls_V(i) < baseline2)
            {
                CorrectValue_2nd_CLF(i) = 1;
            }else
            {
                CorrectValue_2nd_CLF(i) = 0;
            }
        }

        float Accuracy_2ndCls_CLF = (sum(CorrectValue_2nd_CLF)/Classified_2ndCls.n_rows)*100;
        ui->LB_Accur_2ndCls_CLFY->setText(QString::number(Accuracy_2ndCls_CLF, 'f', 2)+" % ");

        //Third Class
        //Convert Predicted_3rdCls to vector
        vec Classified_3rdCls_V = vectorise(Classified_3rdCls);
        vec  CorrectValue_3rd_CLF = zeros<vec>(Classified_3rdCls.n_rows);
        for (uword i = 0; i < Classified_3rdCls.n_rows; i++)
        {
            if(Classified_3rdCls_V(i) > baseline2 && Classified_3rdCls_V(i) < baseline3)
            {
                CorrectValue_3rd_CLF(i) = 1;
            }else
            {
                CorrectValue_3rd_CLF(i) = 0;
            }
        }

        float Accuracy_3rdCls_CLF = (sum(CorrectValue_3rd_CLF)/Classified_3rdCls.n_rows)*100;
        ui->LB_Accur_3rdCls_CLFY->setText(QString::number(Accuracy_3rdCls_CLF, 'f', 2)+" % ");

        //Fourth Class
        //Convert Predicted_3rdCls to vector
        vec Classified_4thCls_V = vectorise(Classified_4thCls);
        vec  CorrectValue_4th_CLF = zeros<vec>(Classified_4thCls.n_rows);
        for (uword i = 0; i < Classified_4thCls.n_rows; i++)
        {
            if(Classified_4thCls_V(i) > baseline3)
            {
                CorrectValue_4th_CLF(i) = 1;
            }else
            {
                CorrectValue_4th_CLF(i) = 0;
            }
        }

        float Accuracy_4thCls_CLF = (sum(CorrectValue_4th_CLF)/Classified_4thCls.n_rows)*100;
        ui->LB_Accur_4thCls_CLFY->setText(QString::number(Accuracy_4thCls_CLF, 'f', 2)+" % ");

        //Overall Accuracy (Average of first, second, third and fourth classes)
        float Overall_Accuracy_CLF = (Accuracy_1stCls_CLF + Accuracy_2ndCls_CLF + Accuracy_3rdCls_CLF + Accuracy_4thCls_CLF)/4;
        ui->LB_Overall_Accur_CLFY->setText(QString::number(Overall_Accuracy_CLF, 'f', 2)+" % ");

    }else if (ui->SP_Classes_CLFY->value() == 5)
    {
        //Third Class
        int F_Index_3rdCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value();
        int L_Index_3rdCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value()-1;
        vec response_3rdCls_CLF = response_data_Pred(span(F_Index_3rdCls_CLF, L_Index_3rdCls_CLF));
        stdvec response_3rdCls_CLF_Cvec = conv_to< stdvec >::from(response_3rdCls_CLF);
        //Convert c++ vectors to qvector
        QVector<double> response_3rdCls_CLF_qvec = QVector<double>(response_3rdCls_CLF_Cvec.begin(), response_3rdCls_CLF_Cvec.end());

        //Classified using Beta Coefficient
        mat Classified_3rdCls = predicted_test(span(F_Index_3rdCls_CLF, L_Index_3rdCls_CLF), 0);
        stdvec Classified_3rdCls_Cvec = conv_to< stdvec >::from(Classified_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> Classified_3rdCls_qvec = QVector<double>(Classified_3rdCls_Cvec.begin(), Classified_3rdCls_Cvec.end());

        //Fourth Class
        int F_Index_4thCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value();
        int L_Index_4thCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value()+ui->SP_4thClass_CLFY->value()-1;
        vec response_4thCls_CLF = response_data_Pred(span(F_Index_4thCls_CLF, L_Index_4thCls_CLF));
        stdvec response_4thCls_CLF_Cvec = conv_to< stdvec >::from(response_4thCls_CLF);
        //Convert c++ vectors to qvector
        QVector<double> response_4thCls_CLF_qvec = QVector<double>(response_4thCls_CLF_Cvec.begin(), response_4thCls_CLF_Cvec.end());

        //Classified using Beta Coefficient
        mat Classified_4thCls = predicted_test(span(F_Index_4thCls_CLF, L_Index_4thCls_CLF), 0);
        stdvec Classified_4thCls_Cvec = conv_to< stdvec >::from(Classified_4thCls);
        //Convert c++ vectors to qvector
        QVector<double> Classified_4thCls_qvec = QVector<double>(Classified_4thCls_Cvec.begin(), Classified_4thCls_Cvec.end());

        //Fifth Class
        //Fifth Class implemented differently (tail)
        int Last_Elements_CLF = ui->SP_5thClass_CLFY->value();
        vec response_5thCls_CLF = response_data_Pred.tail(Last_Elements_CLF);
        stdvec response_5thCls_CLF_Cvec = conv_to< stdvec >::from(response_5thCls_CLF);
        //Convert c++ vectors to qvector
        QVector<double> response_5thCls_CLF_qvec = QVector<double>(response_5thCls_CLF_Cvec.begin(), response_5thCls_CLF_Cvec.end());

        //Classified using Beta Coefficient
        mat Classified_5thCls = predicted_test.tail_rows(Last_Elements_CLF);
        stdvec Classified_5thCls_Cvec = conv_to< stdvec >::from(Classified_5thCls);
        //Convert c++ vectors to qvector
        QVector<double> Classified_5thCls_qvec = QVector<double>(Classified_5thCls_Cvec.begin(), Classified_5thCls_Cvec.end());

        //Add graphs so as First, second, third and Fourth classes can be identified in the legend
        ui->Plt_PredRef_NewD->addGraph();
        ui->Plt_PredRef_NewD->addGraph();
        ui->Plt_PredRef_NewD->addGraph();
        ui->Plt_PredRef_NewD->addGraph();

        //Set range for x-axis
        ui->Plt_PredRef_NewD->xAxis->setRange(-0.5, 4.5);
        ui->Plt_PredRef_NewD->yAxis->setRange(-0.5, 4.5);

        //setup legend
        //Clear existing items first
        ui->Plt_PredRef_NewD->legend->clearItems();
        ui->Plt_PredRef_NewD->replot();

        //Legend for First Class
        ui->Plt_PredRef_NewD->addGraph()->setName("First Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for first class
        ui->Plt_PredRef_NewD->graph(0)->setData(response_1stCls_CLFY_qvec, Classified_1stCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(0)->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plt_PredRef_NewD->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle , Point_thick));
        ui->Plt_PredRef_NewD->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Second Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Second Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Second class
        ui->Plt_PredRef_NewD->graph(1)->setData(response_2ndCls_CLFY_qvec, Classified_2ndCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(1)->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plt_PredRef_NewD->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plt_PredRef_NewD->graph(1)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Third Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Third Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::darkGreen)); // line color darkGreen for Third graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Third class
        ui->Plt_PredRef_NewD->graph(2)->setData(response_3rdCls_CLF_qvec, Classified_3rdCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(2)->setPen(QPen(Qt::darkGreen)); // line color darkGreen for Third graph
        ui->Plt_PredRef_NewD->graph(2)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plt_PredRef_NewD->graph(2)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Fourth Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Fourth Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Fourth class
        ui->Plt_PredRef_NewD->graph(3)->setData(response_4thCls_CLF_qvec, Classified_4thCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(3)->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plt_PredRef_NewD->graph(3)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plt_PredRef_NewD->graph(3)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Fifth Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Fifth Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::darkYellow)); // line color darkYellow for Fifth graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Fifth class
        ui->Plt_PredRef_NewD->graph(4)->setData(response_5thCls_CLF_qvec, Classified_5thCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(4)->setPen(QPen(Qt::darkYellow)); // line color darkYellow for Fifth graph
        ui->Plt_PredRef_NewD->graph(4)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle, Point_thick));
        ui->Plt_PredRef_NewD->graph(4)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        ui->Plt_PredRef_NewD->replot();

        //Calculate Accuracy
        double baseline1 = 0.5;
        double baseline2 = 1.5;
        double baseline3 = 2.5;
        double baseline4 = 3.5;
        //First Class
        //Convert Classified_1stCls to vector
        vec Classified_1stCls_V = vectorise(Classified_1stCls);
        vec  CorrectValue_1st_CLF = zeros<vec>(Classified_1stCls.n_rows);
        for (uword i = 0; i < Classified_1stCls.n_rows; i++)
        {
            if(Classified_1stCls_V(i) < baseline1)
            {
                CorrectValue_1st_CLF(i) = 1;
            }else
            {
                CorrectValue_1st_CLF(i) = 0;
            }
        }

        float Accuracy_1stCls_CLF = (sum(CorrectValue_1st_CLF)/Classified_1stCls.n_rows)*100;
        ui->LB_Accur_1stCls_CLFY->setText(QString::number(Accuracy_1stCls_CLF, 'f', 2)+" % ");

        //Second Class
        //Convert Classified_2ndCls to vector
        vec Classified_2ndCls_V = vectorise(Classified_2ndCls);
        vec  CorrectValue_2nd_CLF = zeros<vec>(Classified_2ndCls.n_rows);
        for (uword i = 0; i < Classified_2ndCls.n_rows; i++)
        {
            if(Classified_2ndCls_V(i) > baseline1 && Classified_2ndCls_V(i) < baseline2)
            {
                CorrectValue_2nd_CLF(i) = 1;
            }else
            {
                CorrectValue_2nd_CLF(i) = 0;
            }
        }

        float Accuracy_2ndCls_CLF = (sum(CorrectValue_2nd_CLF)/Classified_1stCls.n_rows)*100;
        ui->LB_Accur_2ndCls_CLFY->setText(QString::number(Accuracy_2ndCls_CLF, 'f', 2)+" % ");

        //Third Class
        //Convert Classified_3rdCls to vector
        vec Classified_3rdCls_V = vectorise(Classified_3rdCls);
        vec  CorrectValue_3rd_CLF = zeros<vec>(Classified_3rdCls.n_rows);
        for (uword i = 0; i < Classified_3rdCls.n_rows; i++)
        {
            if(Classified_3rdCls_V(i) > baseline2 && Classified_3rdCls_V(i) < baseline3)
            {
                CorrectValue_3rd_CLF(i) = 1;
            }else
            {
                CorrectValue_3rd_CLF(i) = 0;
            }
        }

        float Accuracy_3rdCls_CLF = (sum(CorrectValue_3rd_CLF)/Classified_3rdCls.n_rows)*100;
        ui->LB_Accur_3rdCls_CLFY->setText(QString::number(Accuracy_3rdCls_CLF, 'f', 2)+" % ");

        //Fourth Class
        //Convert Classified_4thCls to vector
        vec Classified_4thCls_V = vectorise(Classified_4thCls);
        vec CorrectValue_4th_CLF = zeros<vec>(Classified_4thCls.n_rows);
        for (uword i = 0; i < Classified_4thCls.n_rows; i++)
        {
            if(Classified_4thCls_V(i) > baseline3 && Classified_4thCls_V(i) < baseline4)
            {
                CorrectValue_4th_CLF(i) = 1;
            }else
            {
                CorrectValue_4th_CLF(i) = 0;
            }
        }

        float Accuracy_4thCls_CLF = (sum(CorrectValue_4th_CLF)/Classified_4thCls.n_rows)*100;
        ui->LB_Accur_4thCls_CLFY->setText(QString::number(Accuracy_4thCls_CLF, 'f', 2)+" % ");

        //Fifth Class
        //Convert Classified_5thCls to vector
        vec Classified_5thCls_V = vectorise(Classified_5thCls);
        vec  CorrectValue_5th_CLF = zeros<vec>(Classified_5thCls.n_rows);
        for (uword i = 0; i < Classified_5thCls.n_rows; i++)
        {
            if(Classified_5thCls_V(i) > baseline4)
            {
                CorrectValue_5th_CLF(i) = 1;
            }else
            {
                CorrectValue_5th_CLF(i) = 0;
            }
        }

        float Accuracy_5thCls_CLF = (sum(CorrectValue_5th_CLF)/Classified_5thCls.n_rows)*100;
        ui->LB_Accur_5thCls_CLFY->setText(QString::number(Accuracy_5thCls_CLF, 'f', 2)+" % ");

        //Overall Accuracy (Average of first, second, third, fourth and fifth classes)
        float Overall_Accuracy_CLF = (Accuracy_1stCls_CLF + Accuracy_2ndCls_CLF + Accuracy_3rdCls_CLF + Accuracy_4thCls_CLF + Accuracy_5thCls_CLF)/5;
        ui->LB_Overall_Accur_CLFY->setText(QString::number(Overall_Accuracy_CLF, 'f', 2)+" % ");

    }else if (ui->SP_Classes_CLFY->value() == 6)
    {
        //Third Class
        int F_Index_3rdCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value();
        int L_Index_3rdCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value()-1;
        vec response_3rdCls_CLF = response_data_Pred(span(F_Index_3rdCls_CLF, L_Index_3rdCls_CLF));
        stdvec response_3rdCls_CLF_Cvec = conv_to< stdvec >::from(response_3rdCls_CLF);
        //Convert c++ vectors to qvector
        QVector<double> response_3rdCls_CLF_qvec = QVector<double>(response_3rdCls_CLF_Cvec.begin(), response_3rdCls_CLF_Cvec.end());

        //Classified using Beta Coefficient
        mat Classified_3rdCls = predicted_test(span(F_Index_3rdCls_CLF, L_Index_3rdCls_CLF), 0);
        stdvec Classified_3rdCls_Cvec = conv_to< stdvec >::from(Classified_3rdCls);
        //Convert c++ vectors to qvector
        QVector<double> Classified_3rdCls_qvec = QVector<double>(Classified_3rdCls_Cvec.begin(), Classified_3rdCls_Cvec.end());

        //Fourth Class
        int F_Index_4thCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value();
        int L_Index_4thCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value()+ui->SP_4thClass_CLFY->value()-1;
        vec response_4thCls_CLF = response_data_Pred(span(F_Index_4thCls_CLF, L_Index_4thCls_CLF));
        stdvec response_4thCls_CLF_Cvec = conv_to< stdvec >::from(response_4thCls_CLF);
        //Convert c++ vectors to qvector
        QVector<double> response_4thCls_CLF_qvec = QVector<double>(response_4thCls_CLF_Cvec.begin(), response_4thCls_CLF_Cvec.end());

        //Classified using Beta Coefficient
        mat Classified_4thCls = predicted_test(span(F_Index_4thCls_CLF, L_Index_4thCls_CLF), 0);
        stdvec Classified_4thCls_Cvec = conv_to< stdvec >::from(Classified_4thCls);
        //Convert c++ vectors to qvector
        QVector<double> Classified_4thCls_qvec = QVector<double>(Classified_4thCls_Cvec.begin(), Classified_4thCls_Cvec.end());

        //Fifth Class
        int F_Index_5thCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value()+ui->SP_4thClass_CLFY->value();
        int L_Index_5thCls_CLF = ui->SP_1stClass_CLFY->value()+ui->SP_2ndClass_CLFY->value()+ui->SP_3rdClass_CLFY->value()+ui->SP_4thClass_CLFY->value()+ui->SP_5thClass_CLFY->value()-1;
        vec response_5thCls_CLF = response_data_Pred(span(F_Index_5thCls_CLF, L_Index_5thCls_CLF));
        stdvec response_5thCls_CLF_Cvec = conv_to< stdvec >::from(response_5thCls_CLF);
        //Convert c++ vectors to qvector
        QVector<double> response_5thCls_CLF_qvec = QVector<double>(response_5thCls_CLF_Cvec.begin(), response_5thCls_CLF_Cvec.end());

        //Classified using Beta Coefficient
        mat Classified_5thCls = predicted_test(span(F_Index_5thCls_CLF, L_Index_5thCls_CLF), 0);
        stdvec Classified_5thCls_Cvec = conv_to< stdvec >::from(Classified_5thCls);
        //Convert c++ vectors to qvector
        QVector<double> Classified_5thCls_qvec = QVector<double>(Classified_5thCls_Cvec.begin(), Classified_5thCls_Cvec.end());

        //Sixth Class
        //Sixth Class implemented differently (tail)
        int Last_Elements_CLF = ui->SP_6thClass_CLFY->value();
        vec response_6thCls_CLF = response_data_Pred.tail(Last_Elements_CLF);
        stdvec response_6thCls_CLF_Cvec = conv_to< stdvec >::from(response_6thCls_CLF);
        //Convert c++ vectors to qvector
        QVector<double> response_6thCls_CLF_qvec = QVector<double>(response_6thCls_CLF_Cvec.begin(), response_6thCls_CLF_Cvec.end());

        //Classified using Beta Coefficient
        mat Classified_6thCls = predicted_test.tail_rows(Last_Elements_CLF);
        stdvec Classified_6thCls_Cvec = conv_to< stdvec >::from(Classified_6thCls);
        //Convert c++ vectors to qvector
        QVector<double> Classified_6thCls_qvec = QVector<double>(Classified_6thCls_Cvec.begin(), Classified_6thCls_Cvec.end());

        //Add graphs so as First, second, third, Fourth and fifth classes can be identified in the legend
        ui->Plt_PredRef_NewD->addGraph();
        ui->Plt_PredRef_NewD->addGraph();
        ui->Plt_PredRef_NewD->addGraph();
        ui->Plt_PredRef_NewD->addGraph();
        ui->Plt_PredRef_NewD->addGraph();

        //Set range for x-axis
        ui->Plt_PredRef_NewD->xAxis->setRange(-0.5, 5.5);
        ui->Plt_PredRef_NewD->yAxis->setRange(-0.5, 5.5);

        //setup legend
        //Clear existing items first
        ui->Plt_PredRef_NewD->legend->clearItems();
        ui->Plt_PredRef_NewD->replot();

        //Legend for First Class
        ui->Plt_PredRef_NewD->addGraph()->setName("First Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for first class
        ui->Plt_PredRef_NewD->graph(0)->setData(response_1stCls_CLFY_qvec, Classified_1stCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(0)->setPen(QPen(Qt::red)); // line color red for first graph
        ui->Plt_PredRef_NewD->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle , Point_thick));
        ui->Plt_PredRef_NewD->graph(0)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Second Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Second Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Second class
        ui->Plt_PredRef_NewD->graph(1)->setData(response_2ndCls_CLFY_qvec, Classified_2ndCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(1)->setPen(QPen(Qt::blue)); // line color blue for second graph
        ui->Plt_PredRef_NewD->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, Point_thick));
        ui->Plt_PredRef_NewD->graph(1)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Third Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Third Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::darkGreen)); // line color darkGreen for Third graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Third class
        ui->Plt_PredRef_NewD->graph(2)->setData(response_3rdCls_CLF_qvec, Classified_3rdCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(2)->setPen(QPen(Qt::darkGreen)); // line color darkGreen for Third graph
        ui->Plt_PredRef_NewD->graph(2)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Point_thick));
        ui->Plt_PredRef_NewD->graph(2)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Fourth Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Fourth Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Fourth class
        ui->Plt_PredRef_NewD->graph(3)->setData(response_4thCls_CLF_qvec, Classified_4thCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(3)->setPen(QPen(Qt::darkMagenta)); // line color darkMagenta for Fourth graph
        ui->Plt_PredRef_NewD->graph(3)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare , Point_thick));
        ui->Plt_PredRef_NewD->graph(3)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Fifth Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Fifth Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::darkYellow)); // line color darkYellow for Fifth graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Fifth class
        ui->Plt_PredRef_NewD->graph(4)->setData(response_5thCls_CLF_qvec, Classified_5thCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(4)->setPen(QPen(Qt::darkYellow)); // line color darkYellow for Fifth graph
        ui->Plt_PredRef_NewD->graph(4)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle, Point_thick));
        ui->Plt_PredRef_NewD->graph(4)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Legend for Sixth Class
        ui->Plt_PredRef_NewD->addGraph()->setName("Sixth Class"); // add new graph
        ui->Plt_PredRef_NewD->graph()->setPen(QPen(Qt::black)); // line color black for sixth graph
        ui->Plt_PredRef_NewD->graph()->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDiamond, Point_thick));
        ui->Plt_PredRef_NewD->graph()->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        //Setup graph for Sixth class
        ui->Plt_PredRef_NewD->graph(5)->setData(response_6thCls_CLF_qvec, Classified_6thCls_qvec); // pass data points to graphs:
        ui->Plt_PredRef_NewD->graph(5)->setPen(QPen(Qt::black)); // line color black for Sixth graph
        ui->Plt_PredRef_NewD->graph(5)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDiamond, Point_thick));
        ui->Plt_PredRef_NewD->graph(5)->setLineStyle((QCPGraph::LineStyle)QCPGraph::lsNone);

        ui->Plt_PredRef_NewD->replot();

        //Calculate Accuracy
        double baseline1 = 0.5;
        double baseline2 = 1.5;
        double baseline3 = 2.5;
        double baseline4 = 3.5;
        double baseline5 = 4.5;
        //First Class
        //Convert Classified_1stCls to vector
        vec Classified_1stCls_V = vectorise(Classified_1stCls);
        vec  CorrectValue_1st_CLF = zeros<vec>(Classified_1stCls.n_rows);
        for (uword i = 0; i < Classified_1stCls.n_rows; i++)
        {
            if(Classified_1stCls_V(i) < baseline1)
            {
                CorrectValue_1st_CLF(i) = 1;
            }else
            {
                CorrectValue_1st_CLF(i) = 0;
            }
        }

        float Accuracy_1stCls_CLF = (sum(CorrectValue_1st_CLF)/Classified_1stCls.n_rows)*100;
        ui->LB_Accur_1stCls_CLFY->setText(QString::number(Accuracy_1stCls_CLF, 'f', 2)+" % ");

        //Second Class
        //Convert Classified_2ndCls to vector
        vec Classified_2ndCls_V = vectorise(Classified_2ndCls);
        vec  CorrectValue_2nd_CLF = zeros<vec>(Classified_2ndCls.n_rows);
        for (uword i = 0; i < Classified_2ndCls.n_rows; i++)
        {
            if(Classified_2ndCls_V(i) > baseline1 && Classified_2ndCls_V(i) < baseline2)
            {
                CorrectValue_2nd_CLF(i) = 1;
            }else
            {
                CorrectValue_2nd_CLF(i) = 0;
            }
        }

        float Accuracy_2ndCls_CLF = (sum(CorrectValue_2nd_CLF)/Classified_1stCls.n_rows)*100;
        ui->LB_Accur_2ndCls_CLFY->setText(QString::number(Accuracy_2ndCls_CLF, 'f', 2)+" % ");

        //Third Class
        //Convert Classified_3rdCls to vector
        vec Classified_3rdCls_V = vectorise(Classified_3rdCls);
        vec  CorrectValue_3rd_CLF = zeros<vec>(Classified_3rdCls.n_rows);
        for (uword i = 0; i < Classified_3rdCls.n_rows; i++)
        {
            if(Classified_3rdCls_V(i) > baseline2 && Classified_3rdCls_V(i) < baseline3)
            {
                CorrectValue_3rd_CLF(i) = 1;
            }else
            {
                CorrectValue_3rd_CLF(i) = 0;
            }
        }

        float Accuracy_3rdCls_CLF = (sum(CorrectValue_3rd_CLF)/Classified_3rdCls.n_rows)*100;
        ui->LB_Accur_3rdCls_CLFY->setText(QString::number(Accuracy_3rdCls_CLF, 'f', 2)+" % ");

        //Fourth Class
        //Convert Classified_4thCls to vector
        vec Classified_4thCls_V = vectorise(Classified_4thCls);
        vec CorrectValue_4th_CLF = zeros<vec>(Classified_4thCls.n_rows);
        for (uword i = 0; i < Classified_4thCls.n_rows; i++)
        {
            if(Classified_4thCls_V(i) > baseline3 && Classified_4thCls_V(i) < baseline4)
            {
                CorrectValue_4th_CLF(i) = 1;
            }else
            {
                CorrectValue_4th_CLF(i) = 0;
            }
        }

        float Accuracy_4thCls_CLF = (sum(CorrectValue_4th_CLF)/Classified_4thCls.n_rows)*100;
        ui->LB_Accur_4thCls_CLFY->setText(QString::number(Accuracy_4thCls_CLF, 'f', 2)+" % ");

        //Fifth Class
        //Convert Classified_5thCls to vector
        vec Classified_5thCls_V = vectorise(Classified_5thCls);
        vec  CorrectValue_5th_CLF = zeros<vec>(Classified_5thCls.n_rows);
        for (uword i = 0; i < Classified_5thCls.n_rows; i++)
        {
            if(Classified_5thCls_V(i) > baseline4 && Classified_5thCls_V(i) < baseline5)
            {
                CorrectValue_5th_CLF(i) = 1;
            }else
            {
                CorrectValue_5th_CLF(i) = 0;
            }
        }

        float Accuracy_5thCls_CLF = (sum(CorrectValue_5th_CLF)/Classified_5thCls.n_rows)*100;
        ui->LB_Accur_5thCls_CLFY->setText(QString::number(Accuracy_5thCls_CLF, 'f', 2)+" % ");

        //Sixth Class
        //Convert Classified_6thCls to vector
        vec Classified_6thCls_V = vectorise(Classified_6thCls);
        vec  CorrectValue_6th_CLF = zeros<vec>(Classified_6thCls.n_rows);
        for (uword i = 0; i < Classified_6thCls.n_rows; i++)
        {
            if(Classified_6thCls_V(i) > baseline5)
            {
                CorrectValue_6th_CLF(i) = 1;
            }else
            {
                CorrectValue_6th_CLF(i) = 0;
            }
        }

        float Accuracy_6thCls_CLF = (sum(CorrectValue_6th_CLF)/Classified_6thCls.n_rows)*100;
        ui->LB_Accur_6thCls_CLFY->setText(QString::number(Accuracy_6thCls_CLF, 'f', 2)+" % ");

        //Overall Accuracy (Average of first, second, third, fourth, fifth and sixth classes)
        float Overall_Accuracy_CLF = (Accuracy_1stCls_CLF + Accuracy_2ndCls_CLF + Accuracy_3rdCls_CLF + Accuracy_4thCls_CLF + Accuracy_5thCls_CLF + Accuracy_6thCls_CLF)/6;
        ui->LB_Overall_Accur_CLFY->setText(QString::number(Overall_Accuracy_CLF, 'f', 2)+" % ");

    }else{
        QMessageBox::information(this, "Warning", "Number of Classes not Currently Supported");
    }
}

//Save Predicted vs Reference plot for PLSR Training model - regression
void MainWindow::on_PB_Save_PredRef_clicked()
{
    //Use filters for file extensions
    QString filter = "All Files (*.*);; png Files (*.png);; pdf Files (*.pdf);; CSV Files (*.csv);; jpg Files (*.jpg);; "
                    "jpeg Files (*.jpeg);; bmp Files (*.bmp);; Rastered PNG Files (*.PNG);; Rastered JPG Files (*.JPG);; "
                    "Rastered JPEG Files (*.JPEG);; Rastered BMP Files (*.BMP)";

    //Open dialog for saving
    QString fileName = QFileDialog::getSaveFileName(this, "Save Predicted vs Reference Plot", qApp->applicationDirPath(), filter);

    //Confirm file extensions and save
    if (fileName.endsWith(".png"))
    {
        ui->Plot_PredVRef->savePng(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".jpg"))
    {
        ui->Plot_PredVRef->saveJpg(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".jpeg"))
    {
        ui->Plot_PredVRef->saveJpg(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".bmp"))
    {
        ui->Plot_PredVRef->saveBmp(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".pdf"))
    {
        ui->Plot_PredVRef->savePdf(fileName, 891, 630, QCP::epNoCosmetic);
    }
    else if (fileName.endsWith(".PNG"))
    {
        ui->Plot_PredVRef->saveRastered(fileName, 891, 630, 2.0, "PNG");
    }
    else if (fileName.endsWith(".JPG"))
    {
        ui->Plot_PredVRef->saveRastered(fileName, 891, 630, 2.0, "JPG");
    }
    else if (fileName.endsWith(".BMP"))
    {
        ui->Plot_PredVRef->saveRastered(fileName, 891, 630, 2.0, "BMP");
    }
    else if (fileName.endsWith(".JPEG"))
    {
        ui->Plot_PredVRef->saveRastered(fileName, 891, 630, 2.0, "JPEG");
    }else if (fileName.endsWith(".csv"))
    {
        //Join the Predicted and reference matrices together - join horizontally
        mat Combine_Ref_Pred_PLSR = join_rows(response_data,Predicted_optm);

        // save in CSV format without a header
        Combine_Ref_Pred_PLSR.save(fileName.toStdString(), csv_ascii);

    }
    else
    {
        QMessageBox::warning(this, "Attention", "Specify the format");
    }
}

//Save Predicted vs Reference plot for PLS-DA Training model - Classification
void MainWindow::on_PB_Sve_PredRef_Class_clicked()
{
    //Use filters for file extensions
    QString filter = "All Files (*.*);; png Files (*.png);; pdf Files (*.pdf);; CSV Files (*.csv);; jpg Files (*.jpg);; "
                    "jpeg Files (*.jpeg);; bmp Files (*.bmp);; Rastered PNG Files (*.PNG);; Rastered JPG Files (*.JPG);; "
                    "Rastered JPEG Files (*.JPEG);; Rastered BMP Files (*.BMP)";

    //Open dialog for saving
    QString fileName = QFileDialog::getSaveFileName(this, "Save Predicted vs Reference Plot", qApp->applicationDirPath(), filter);

    //Confirm file extensions and save
    if (fileName.endsWith(".png"))
    {
        ui->Plot_PredVRef->savePng(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".jpg"))
    {
        ui->Plot_PredVRef->saveJpg(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".jpeg"))
    {
        ui->Plot_PredVRef->saveJpg(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".bmp"))
    {
        ui->Plot_PredVRef->saveBmp(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".pdf"))
    {
        ui->Plot_PredVRef->savePdf(fileName, 891, 630, QCP::epNoCosmetic);
    }
    else if (fileName.endsWith(".PNG"))
    {
        ui->Plot_PredVRef->saveRastered(fileName, 891, 630, 2.0, "PNG");
    }
    else if (fileName.endsWith(".JPG"))
    {
        ui->Plot_PredVRef->saveRastered(fileName, 891, 630, 2.0, "JPG");
    }
    else if (fileName.endsWith(".BMP"))
    {
        ui->Plot_PredVRef->saveRastered(fileName, 891, 630, 2.0, "BMP");
    }
    else if (fileName.endsWith(".JPEG"))
    {
        ui->Plot_PredVRef->saveRastered(fileName, 891, 630, 2.0, "JPEG");
    }else if (fileName.endsWith(".csv"))
    {
        //Join the Predicted and reference matrices together - join horizontally
        mat Combine_Ref_Pred_PLSDA = join_rows(response_data,Predicted_optm);

        // save in CSV format without a header
        Combine_Ref_Pred_PLSDA.save(fileName.toStdString(), csv_ascii);

    }
    else
    {
        QMessageBox::warning(this, "Attention", "Specify the format");
    }
}

//Save the beta Coefficient - same plot for both PLSR and PLS-DA Model (PLS Model)
void MainWindow::on_PB_Save_Beta_clicked()
{
    //Use filters for file extensions
    QString filter = "All Files (*.*);; png Files (*.png);; pdf Files (*.pdf);; CSV Files (*.csv);; jpg Files (*.jpg);; "
                    "jpeg Files (*.jpeg);; bmp Files (*.bmp);; Rastered PNG Files (*.PNG);; Rastered JPG Files (*.JPG);; "
                    "Rastered JPEG Files (*.JPEG);; Rastered BMP Files (*.BMP)";

    //Open dialog for saving
    QString fileName = QFileDialog::getSaveFileName(this, "Save Predicted vs Reference Plot", qApp->applicationDirPath(), filter);

    //Confirm file extensions and save
    if (fileName.endsWith(".png"))
    {
        ui->Plot_Beta->savePng(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".jpg"))
    {
        ui->Plot_Beta->saveJpg(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".jpeg"))
    {
        ui->Plot_Beta->saveJpg(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".bmp"))
    {
        ui->Plot_Beta->saveBmp(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".pdf"))
    {
        ui->Plot_Beta->savePdf(fileName, 891, 630, QCP::epNoCosmetic);
    }
    else if (fileName.endsWith(".PNG"))
    {
        ui->Plot_Beta->saveRastered(fileName, 891, 630, 2.0, "PNG");
    }
    else if (fileName.endsWith(".JPG"))
    {
        ui->Plot_Beta->saveRastered(fileName, 891, 630, 2.0, "JPG");
    }
    else if (fileName.endsWith(".BMP"))
    {
        ui->Plot_Beta->saveRastered(fileName, 891, 630, 2.0, "BMP");
    }
    else if (fileName.endsWith(".JPEG"))
    {
        ui->Plot_Beta->saveRastered(fileName, 891, 630, 2.0, "JPEG");
    }else if (fileName.endsWith(".csv"))
    {
        // save in CSV format without a header
        Beta_optm.save(fileName.toStdString(), csv_ascii);

    }
    else
    {
        QMessageBox::warning(this, "Attention", "Specify the format");
    }
}

//Save Predicted vs Reference plot for New dataset - predicted using PLSR trained model
void MainWindow::on_PB_Save_PredRef_NewD_clicked()
{
    //Use filters for file extensions
    QString filter = "All Files (*.*);; png Files (*.png);; pdf Files (*.pdf);; CSV Files (*.csv);; jpg Files (*.jpg);; "
                    "jpeg Files (*.jpeg);; bmp Files (*.bmp);; Rastered PNG Files (*.PNG);; Rastered JPG Files (*.JPG);; "
                    "Rastered JPEG Files (*.JPEG);; Rastered BMP Files (*.BMP)";

    //Open dialog for saving
    QString fileName = QFileDialog::getSaveFileName(this, "Save Predicted vs Reference Plot", qApp->applicationDirPath(), filter);

    //Confirm file extensions and save
    if (fileName.endsWith(".png"))
    {
        ui->Plt_PredRef_NewD->savePng(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".jpg"))
    {
        ui->Plt_PredRef_NewD->saveJpg(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".jpeg"))
    {
        ui->Plt_PredRef_NewD->saveJpg(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".bmp"))
    {
        ui->Plt_PredRef_NewD->saveBmp(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".pdf"))
    {
        ui->Plt_PredRef_NewD->savePdf(fileName, 891, 630, QCP::epNoCosmetic);
    }
    else if (fileName.endsWith(".PNG"))
    {
        ui->Plt_PredRef_NewD->saveRastered(fileName, 891, 630, 2.0, "PNG");
    }
    else if (fileName.endsWith(".JPG"))
    {
        ui->Plt_PredRef_NewD->saveRastered(fileName, 891, 630, 2.0, "JPG");
    }
    else if (fileName.endsWith(".BMP"))
    {
        ui->Plt_PredRef_NewD->saveRastered(fileName, 891, 630, 2.0, "BMP");
    }
    else if (fileName.endsWith(".JPEG"))
    {
        ui->Plt_PredRef_NewD->saveRastered(fileName, 891, 630, 2.0, "JPEG");
    }else if (fileName.endsWith(".csv"))
    {
        //Join the Predicted and reference matrices together - join horizontally
        mat Combine_Ref_Pred_PLSR_NewD = join_rows(response_data_Pred,predicted_test);

        // save in CSV format without a header
        Combine_Ref_Pred_PLSR_NewD.save(fileName.toStdString(), csv_ascii);

    }
    else
    {
        QMessageBox::warning(this, "Attention", "Specify the format");
    }
}

//Save Predicted vs Reference plot for New dataset - predicted using PLSDA (Classification) trained model
void MainWindow::on_PB_Sve_PredRef_Class_CLFY_clicked()
{
    //Use filters for file extensions
    QString filter = "All Files (*.*);; png Files (*.png);; pdf Files (*.pdf);; CSV Files (*.csv);; jpg Files (*.jpg);; "
                    "jpeg Files (*.jpeg);; bmp Files (*.bmp);; Rastered PNG Files (*.PNG);; Rastered JPG Files (*.JPG);; "
                    "Rastered JPEG Files (*.JPEG);; Rastered BMP Files (*.BMP)";

    //Open dialog for saving
    QString fileName = QFileDialog::getSaveFileName(this, "Save Predicted vs Reference Plot", qApp->applicationDirPath(), filter);

    //Confirm file extensions and save
    if (fileName.endsWith(".png"))
    {
        ui->Plt_PredRef_NewD->savePng(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".jpg"))
    {
        ui->Plt_PredRef_NewD->saveJpg(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".jpeg"))
    {
        ui->Plt_PredRef_NewD->saveJpg(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".bmp"))
    {
        ui->Plt_PredRef_NewD->saveBmp(fileName, 891, 630, 2.0);
    }
    else if (fileName.endsWith(".pdf"))
    {
        ui->Plt_PredRef_NewD->savePdf(fileName, 891, 630, QCP::epNoCosmetic);
    }
    else if (fileName.endsWith(".PNG"))
    {
        ui->Plt_PredRef_NewD->saveRastered(fileName, 891, 630, 2.0, "PNG");
    }
    else if (fileName.endsWith(".JPG"))
    {
        ui->Plt_PredRef_NewD->saveRastered(fileName, 891, 630, 2.0, "JPG");
    }
    else if (fileName.endsWith(".BMP"))
    {
        ui->Plt_PredRef_NewD->saveRastered(fileName, 891, 630, 2.0, "BMP");
    }
    else if (fileName.endsWith(".JPEG"))
    {
        ui->Plt_PredRef_NewD->saveRastered(fileName, 891, 630, 2.0, "JPEG");
    }else if (fileName.endsWith(".csv"))
    {
        //Join the Predicted and reference matrices together - join horizontally
        mat Combine_Ref_Pred_PLSDA_NewD = join_rows(response_data_Pred,predicted_test);

        // save in CSV format without a header
        Combine_Ref_Pred_PLSDA_NewD.save(fileName.toStdString(), csv_ascii);

    }
    else
    {
        QMessageBox::warning(this, "Attention", "Specify the format");
    }
}

//Open a folder on your computer
void MainWindow::on_actionOpen_triggered()
{
    //Create string for dynamic path
    QString filename1 = QFileDialog::getOpenFileName(this, "Load data from text files", qApp->applicationDirPath());

    QMessageBox::information(this, "PLS Model", "Click on Tasks to train and Validate PLS Model");
}

void MainWindow::on_RB_Savitzky_clicked()
{
    QMessageBox::information(this, "Required Parameters:", "Input the Filter Window size (Odd Number), Derivative and Polynomial Orders");

    //Show the qlabels and Spin boxes for Savitzky Golay when radio button is checked
    //Qlabel
    ui->LB_SG_Derivative->setVisible(true);
    ui->LB_SG_Polynomial->setVisible(true);
    ui->LB_SG_Window->setVisible(true);
    //Qspinbox
    ui->SB_SG_Derivative->setVisible(true);
    ui->SB_SG_Polynomial->setVisible(true);
    ui->SB_SG_Window->setVisible(true);

}

void MainWindow::on_RB_Smooth_clicked()
{
    QMessageBox::information(this, "Required Parameter:", "Input the Filter Window size (Odd Number)");
    //Show the qlabel and Spin box for Moving average filter(smoothing) preprocessing method when radio button is checked
    //Qlabel
    ui->LB_Smooth_Window->setVisible(true);

    //Qspinbox
    ui->SB_Smooth_Window->setVisible(true);
}

void MainWindow::on_RB_Savitzky_Pred_clicked()
{
    QMessageBox::information(this, "Required Parameters:", "Input the same parameter values used in training the model");

    //Show the qlabels and Spin boxes for Savitzky Golay preprocessing method (Prediction) when radio button is checked
    //Qlabel
    ui->LB_SG_Deriv_Pred->setVisible(true);
    ui->LB_SG_Poly_Pred->setVisible(true);
    ui->LB_SG_Wndw_Pred->setVisible(true);
    //Qspinbox
    ui->SB_SG_Deriv_Pred->setVisible(true);
    ui->SB_SG_Poly_Pred->setVisible(true);
    ui->SB_SG_Wndw_Pred->setVisible(true);

}

void MainWindow::on_RB_Smooth_Pred_clicked()
{
    QMessageBox::information(this, "Required Parameter:", "Input the same Filter window size used in training the model");

    //Show the qlabel and Spin box for Moving average filter(smoothing) preprocessing method (Prediction) when radio button is checked
    //Qlabel
    ui->LB_Smooth_Wndw_Pred->setVisible(true);

    //Qspinbox
    ui->SB_Smooth_Wndw_Pred->setVisible(true);
}

void MainWindow::on_PB_PlotSpectra_clicked()
{
    //Pointer for the sec dialog (Plt_Spectra) class
    pltspectra = new Plt_Spectra; //from pointer in mainwindow.h

    mat spectra_data_prepplt;
    //mat spectra_data_prep;
    //Compute the preprocessed predictor (X) data if the radio buttons are checked
    //SNV
    if (ui->RB_SNV->isChecked())
    {
        //Perform SNV using coded function
        mat Spectra_SNVplt;
        Standard_Normal_variate(spectra_data, Spectra_SNVplt);

        //Save MSC processed data
        Spectra_SNVplt.save("SNV_processed.csv", arma::csv_ascii);

        spectra_data_prepplt = Spectra_SNVplt;

    }

    //Multiplicative scatter correction (MSC)
    if (ui->RB_MSC->isChecked())
    {
        //Transpose Input spectra_data so as to perform MSC with Mlpack
        mat input = trans(spectra_data);

        //Perform MSC
        mat XMSC;
        MSC_Process(input, XMSC);

        //Transpose the data to obtain the original orientation of the data after MSC preprocessing
        mat XMSC_transpose = trans(XMSC);

        //Save MSC processed data
        XMSC_transpose.save("MSC_processed.csv", arma::csv_ascii);

        spectra_data_prepplt = XMSC_transpose;

    }

    //Savitzky Golay
    if (ui->RB_Savitzky->isChecked())
    {
        //Perform Savitzky-Golay smoothing/derivatization on the spectral data.
        //Transpose - carry out Savitzky-Golay smoothing/derivatization and transpose again
        mat data_t = trans(spectra_data);
        mat  SG_data_t = zeros<mat>(spectra_data.n_cols, spectra_data.n_rows);
        //polynomial_order The order of the polynomial
        //arma::uword polynomial_order = 3;
        arma::uword polynomial_order = ui->SB_SG_Polynomial->value();

        //window_size The size of the filter window.
        arma::uword window_size = ui->SB_SG_Window->value();

        //derivative_order The order of the derivative.
        arma::uword derivative_order = ui->SB_SG_Derivative->value();
        SG_data_t = sgolayfilt(data_t,polynomial_order,window_size,derivative_order,1);

        mat SG_data = trans(SG_data_t);

        //Save Savitzky-Golay data
        SG_data.save("Savitzky_Golay_data.csv", arma::csv_ascii);

        spectra_data_prepplt = SG_data;

    }

    //Smoothing (Moving Average Filter)
    if (ui->RB_Smooth->isChecked())
    {
        //CreateMovingAverageFilter
        arma::uword window_size = ui->SB_Smooth_Window->value();
        vec filter = CreateMovingAverageFilter(window_size);

        // Perform moving average filtering on the spectral data.
        //Transpose - carry out smoothing and transpose again
        mat data_t = trans(spectra_data);
        mat  Smooth_data_t = zeros<mat>(spectra_data.n_cols, spectra_data.n_rows);
        for (uword j = 0; j < data_t.n_cols; ++j) {
            Smooth_data_t.col(j) = ApplyFilter(data_t.col(j), filter);
        }
        mat Smooth_data = trans(Smooth_data_t);

        //Save smoothed data
        Smooth_data.save("Smoothed_data.csv", arma::csv_ascii);

        spectra_data_prepplt = Smooth_data;

    }

    //Mean normalization
    if (ui->RB_MeanNorm->isChecked())
    {
        //Perform Mean normalization
        //Transpose - carry out Mean normalization and transpose again
        mat spectra_data_t = trans(spectra_data);
        mat Mean_Normalized;
        Mean_Normalize(spectra_data_t, Mean_Normalized);

        //Save mean normalized data
        Mean_Normalized.save("Mean_Normalized_data.csv", arma::csv_ascii);

        spectra_data_prepplt = Mean_Normalized;

    }

    //Min/Max Normalization
    if (ui->RB_MinMaxNorm->isChecked())
    {
        //Perform Min-Max Normalization
        //Transpose - carry out Min-Max Normalization and transpose again
        mat spectra_data_t = trans(spectra_data);
        mat Min_Max_Normalized;
        Min_Max_Normalize(spectra_data_t, Min_Max_Normalized);

        //Save mean normalized data
        Min_Max_Normalized.save("Min_Max_Normalized_data.csv", arma::csv_ascii);

        spectra_data_prepplt = Min_Max_Normalized;

    }

    //Standardization
    if (ui->RB_Standardize->isChecked())
    {
        //Perform Standardization on data
        //Transpose - carry out Standardization and transpose again
        mat spectra_data_t = trans(spectra_data);
        arma::mat Standardized_data;
        Standardization(spectra_data_t, Standardized_data);

        //Save Standardized data
        Standardized_data.save("Standardized_data.csv", arma::csv_ascii);

        spectra_data_prepplt = Standardized_data;

    }

    //None - No preprocessing
    if (ui->RB_NoPreproc->isChecked())
    {
        spectra_data_prepplt = spectra_data;

    }

    //Setup spectra Line graph against the wavelength
    // configure right and top axis to show ticks but no labels:
    pltspectra->ui->Plot_Spectra->axisRect()->setupFullAxesBox();

    //Thickness of lines/curves
    const int pen_width = 1;
    const int pen_width_mean = 6;

    //Axes
    pltspectra->ui->Plot_Spectra->xAxis->setLabel("Wavelength (nm)");
    pltspectra->ui->Plot_Spectra->yAxis->setLabel("Log(1/R)");

    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    pltspectra->ui->Plot_Spectra->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

    //Obtain mean of samples (row vector running through all the wavelength)
    arma::vec mean_spectra = trans(mean(spectra_data_prepplt));

    //Convert armadillo vector to std c++ vector
    stdvec Mean_cvec = conv_to< stdvec >::from(mean_spectra);
    //Convert c++ vectors to qvector
    QVector<double> Mean_qvec = QVector<double>(Mean_cvec.begin(), Mean_cvec.end());

    //plot the mean spectra
    pltspectra->ui->Plot_Spectra->addGraph(); // add new graph
    pltspectra->ui->Plot_Spectra->graph(0)->setName("Log(1/R)"); //Set name
    pltspectra->ui->Plot_Spectra->graph(0)->setData(Wavelength_qvec, Mean_qvec); // pass data points to graphs:
    pltspectra->ui->Plot_Spectra->graph(0)->setPen(QPen(Qt::red, pen_width_mean));
    pltspectra->ui->Plot_Spectra->graph(0)->setLineStyle(QCPGraph::lsLine);
    pltspectra->ui->Plot_Spectra->graph(0)->rescaleAxes();  // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
    pltspectra->ui->Plot_Spectra->replot();

    //Prepare Preprocessed Data for Plotting
    for (uword i = 1; i < spectra_data_prepplt.n_rows; i++)
    {
        //Obtain each vector running through all spectra
        arma::vec Preprocessed_vec = trans(spectra_data_prepplt.row(i));

        //Convert armadillo vector to std c++ vector
        stdvec Preprocessed_cvec = conv_to< stdvec >::from(Preprocessed_vec);
        //Convert c++ vectors to qvector
        QVector<double> Preprocessed_qvec = QVector<double>(Preprocessed_cvec.begin(), Preprocessed_cvec.end());

        //plot the spectra
        pltspectra->ui->Plot_Spectra->addGraph(); // add new graph
        pltspectra->ui->Plot_Spectra->graph(i)->setName("Log(1/R)"); //Set name
        pltspectra->ui->Plot_Spectra->graph(i)->setData(Wavelength_qvec, Preprocessed_qvec); // pass data points to graphs:
        pltspectra->ui->Plot_Spectra->graph(i)->setPen(QPen(Qt::blue, pen_width));
        pltspectra->ui->Plot_Spectra->graph(i)->setLineStyle(QCPGraph::lsLine);
        pltspectra->ui->Plot_Spectra->graph(i)->rescaleAxes();  // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
        pltspectra->ui->Plot_Spectra->replot();

    }

    //Opening second window with access to first window - modaless approach
    //pltspectra = new Plt_Spectra; //from pointer in mainwindow.h
    pltspectra->show();


}

void MainWindow::on_pushButton_clicked()
{
    //Create filter for text files (.txt) as input
    QString filter3 = "Text Files(*.txt);; CSV Files (*.csv)";

    //Create string for dynamic path
    QString filename3 = QFileDialog::getOpenFileName(this, "Load the Wavelength Data from text or CSV files", qApp->applicationDirPath(), filter3);

    //testing for success
    arma::vec Wavelength;
    bool Loaded_file = Wavelength.load(filename3.toStdString());;

    if(Loaded_file == false)
      {
        qDebug() << "Failed....Input a text or csv file";
      }

    //Wavelength.print("Wavelength:");

    //Prepare wavelength Data for Plotting
    //Convert armadillo matrix or vector to std c++ vector
    stdvec Wavelength_cvec = conv_to< stdvec >::from(Wavelength);
    //Convert c++ vectors to qvector
    Wavelength_qvec = QVector<double>(Wavelength_cvec.begin(), Wavelength_cvec.end());

    QMessageBox::information(this, "Status", "Wavelength data Loaded.");
}

void MainWindow::on_PB_plot_Spectra_Pred_clicked()
{
    //Pointer for the sec dialog (Plt_Spectra) class
    pltspectra = new Plt_Spectra; //from pointer in mainwindow.h

    mat spectra_data_Validtn_prep_plt;

    //Compute the preprocessed Validation X data if the radio buttons are checked
    //SNV
    if (ui->RB_SNV_Pred->isChecked())
    {
        //Perform SNV on validation X data
        mat Spectra_SNV_Pred;
        Standard_Normal_variate(spectra_data_Pred, Spectra_SNV_Pred);

        //Save SNV processed data
        Spectra_SNV_Pred.save("SNV_processed_Pred.csv", arma::csv_ascii);

        spectra_data_Validtn_prep_plt = Spectra_SNV_Pred;

    }

    //Multiplicative scatter correction (MSC)
    if (ui->RB_MSC_Pred->isChecked())
    {
        //Transpose Input spectra_data so as to perform MSC with Mlpack
        mat input_Pred = trans(spectra_data_Pred);

        //Perform MSC
        mat XMSC_Pred;
        MSC_Process(input_Pred, XMSC_Pred);

        //Transpose the data to obtain the original orientation of the data after MSC preprocessing
        mat XMSC_trans_Pred = trans(XMSC_Pred);

        //Save MSC processed data
        XMSC_trans_Pred.save("MSC_processed_Pred.csv", arma::csv_ascii);

        spectra_data_Validtn_prep_plt = XMSC_trans_Pred;

    }

    //Savitzky Golay
    if (ui->RB_Savitzky_Pred->isChecked())
    {
        //Perform Savitzky-Golay smoothing/derivatization on the spectral data.
        //Transpose - carry out Savitzky-Golay smoothing/derivatization and transpose again
        mat data_t_Pred = trans(spectra_data_Pred);
        mat  SG_data_t_Pred = zeros<mat>(spectra_data_Pred.n_cols, spectra_data_Pred.n_rows);

        //Polynomial order
        arma::uword poly_order_Pred = ui->SB_SG_Poly_Pred->value();

        //window_size The size of the filter window.
        arma::uword windw_size_Pred = ui->SB_SG_Wndw_Pred->value();

        //derivative_order The order of the derivative.
        arma::uword deriv_order_Pred = ui->SB_SG_Deriv_Pred->value();
        SG_data_t_Pred = sgolayfilt(data_t_Pred,poly_order_Pred,windw_size_Pred,deriv_order_Pred,1);

        mat SG_data_Pred = trans(SG_data_t_Pred);

        //Save Savitzky-Golay data
        SG_data_Pred.save("Savitzky_Golay_data_Pred.csv", arma::csv_ascii);

        spectra_data_Validtn_prep_plt = SG_data_Pred;

    }

    //Smoothing (Moving Average Filter)
    if (ui->RB_Smooth_Pred->isChecked())
    {
        //CreateMovingAverageFilter
        arma::uword wndow_size_Pred = ui->SB_Smooth_Wndw_Pred->value();
        vec filter_Pred = CreateMovingAverageFilter(wndow_size_Pred);

        // Perform moving average filtering on the spectral data.
        //Transpose - carry out smoothing and transpose again
        mat data_t_Pred = trans(spectra_data_Pred);
        mat  Smooth_data_t_P = zeros<mat>(spectra_data_Pred.n_cols, spectra_data_Pred.n_rows);
        for (uword j = 0; j < data_t_Pred.n_cols; ++j) {
            Smooth_data_t_P.col(j) = ApplyFilter(data_t_Pred.col(j), filter_Pred);
        }
        mat Smooth_data_Pred = trans(Smooth_data_t_P);

        //Save smoothed data
        Smooth_data_Pred.save("Smoothed_data_Pred.csv", arma::csv_ascii);

        spectra_data_Validtn_prep_plt = Smooth_data_Pred;

    }

    //Mean normalization
    if (ui->RB_MeanNorm_Pred->isChecked())
    {
        //Perform Mean normalization
        //Transpose - carry out Mean normalization and transpose again
        mat spectra_data_t_P = trans(spectra_data_Pred);
        mat Mean_Normalized_Pred;
        Mean_Normalize(spectra_data_t_P, Mean_Normalized_Pred);

        //Save mean normalized data
        Mean_Normalized_Pred.save("Mean_Normalized_data_Pred.csv", arma::csv_ascii);

        spectra_data_Validtn_prep_plt = Mean_Normalized_Pred;

    }

    //Min/Max Normalization
    if (ui->RB_MinMaxNorm_Pred->isChecked())
    {
        //Perform Min-Max Normalization
        //Transpose - carry out Min-Max Normalization and transpose again
        mat spectra_data_t_P = trans(spectra_data_Pred);
        mat Min_Max_Normalized_P;
        Min_Max_Normalize(spectra_data_t_P, Min_Max_Normalized_P);

        //Save mean normalized data
        Min_Max_Normalized_P.save("Min_Max_Normalized_data_P.csv", arma::csv_ascii);

        spectra_data_Validtn_prep_plt = Min_Max_Normalized_P;

    }

    //Standardization
    if (ui->RB_Standard_Pred->isChecked())
    {
        //Perform Standardization on data
        //Transpose - carry out Standardization and transpose again
        mat spectra_data_t_P = trans(spectra_data_Pred);
        arma::mat Standardized_data_P;
        Standardization(spectra_data_t_P, Standardized_data_P);

        //Save Standardized data
        Standardized_data_P.save("Standardized_data_P.csv", arma::csv_ascii);

        spectra_data_Validtn_prep_plt = Standardized_data_P;

    }

    //None - No preprocessing
    if (ui->RB_NoPreproc_Pred->isChecked())
    {
        spectra_data_Validtn_prep_plt = spectra_data_Pred;

    }

    //Setup spectra Line graph against the wavelength
    // configure right and top axis to show ticks but no labels:
    pltspectra->ui->Plot_Spectra->axisRect()->setupFullAxesBox();

    //Thickness of lines/curves
    const int pen_width = 1;
    const int pen_width_mean = 6;

    //Axes
    pltspectra->ui->Plot_Spectra->xAxis->setLabel("Wavelength (nm)");
    pltspectra->ui->Plot_Spectra->yAxis->setLabel("Log(1/R)");

    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    pltspectra->ui->Plot_Spectra->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

    //Obtain mean of samples (row vector running through all the wavelength)
    arma::vec mean_spectra = trans(mean(spectra_data_Validtn_prep_plt));

    //Convert armadillo vector to std c++ vector
    stdvec Mean_cvec = conv_to< stdvec >::from(mean_spectra);
    //Convert c++ vectors to qvector
    QVector<double> Mean_qvec = QVector<double>(Mean_cvec.begin(), Mean_cvec.end());

    //plot the mean spectra
    pltspectra->ui->Plot_Spectra->addGraph(); // add new graph
    pltspectra->ui->Plot_Spectra->graph(0)->setName("Log(1/R)"); //Set name
    pltspectra->ui->Plot_Spectra->graph(0)->setData(Wavelength_qvec, Mean_qvec); // pass data points to graphs:
    pltspectra->ui->Plot_Spectra->graph(0)->setPen(QPen(Qt::red, pen_width_mean));
    pltspectra->ui->Plot_Spectra->graph(0)->setLineStyle(QCPGraph::lsLine);
    pltspectra->ui->Plot_Spectra->graph(0)->rescaleAxes();  // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
    pltspectra->ui->Plot_Spectra->replot();

    //Prepare Preprocessed Data for Plotting
    for (uword i = 1; i < spectra_data_Validtn_prep_plt.n_rows; i++)
    {
        //Obtain each vector running through all spectra
        arma::vec Preprocessed_vec_pred = trans(spectra_data_Validtn_prep_plt.row(i));

        //Convert armadillo vector to std c++ vector
        stdvec Preprocessed_cvec_pred = conv_to< stdvec >::from(Preprocessed_vec_pred);
        //Convert c++ vectors to qvector
        QVector<double> Preprocessed_qvec_pred = QVector<double>(Preprocessed_cvec_pred.begin(), Preprocessed_cvec_pred.end());

        //plot the spectra
        pltspectra->ui->Plot_Spectra->addGraph(); // add new graph
        pltspectra->ui->Plot_Spectra->graph(i)->setName("Log(1/R)"); //Set name
        pltspectra->ui->Plot_Spectra->graph(i)->setData(Wavelength_qvec, Preprocessed_qvec_pred); // pass data points to graphs:
        pltspectra->ui->Plot_Spectra->graph(i)->setPen(QPen(Qt::blue, pen_width));
        pltspectra->ui->Plot_Spectra->graph(i)->setLineStyle(QCPGraph::lsLine);
        pltspectra->ui->Plot_Spectra->graph(i)->rescaleAxes();  // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
        pltspectra->ui->Plot_Spectra->replot();

    }

    //Opening second window with access to first window - modaless approach
    //pltspectra = new Plt_Spectra; //from pointer in mainwindow.h
    pltspectra->show();

}

void MainWindow::on_actionAbout_triggered()
{
    //Open widget with about details
    ui->stackedWidget->setCurrentIndex(5);
}
