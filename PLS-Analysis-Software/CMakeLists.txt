cmake_minimum_required(VERSION 3.5)

project(PLS_Analysis_GUI LANGUAGES CXX)
#project(PLS_Analysis_GUI LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check https://doc.qt.io/qt/deployment-android.html for more information.
# They need to be set before the find_package( ...) calls below.

#if(ANDROID)
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    if (ANDROID_ABI STREQUAL "armeabi-v7a")
#        set(ANDROID_EXTRA_LIBS
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
#    endif()
#endif()

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets PrintSupport REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets PrintSupport REQUIRED)

#Find Armadillo
find_package(Armadillo CONFIG REQUIRED)

#Find MLPACK
find_package(mlpack REQUIRED)

set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
        Resources.qrc
        qcustomplot
        plt_spectra.cpp
        plt_spectra.h
        plt_spectra.ui
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(PLS_Analysis_GUI
        ${PROJECT_SOURCES}
    )
else()
    if(ANDROID)
        add_library(PLS_Analysis_GUI SHARED
            ${PROJECT_SOURCES}
        )
    else()
        add_executable(PLS_Analysis_GUI
            ${PROJECT_SOURCES}
        )
    endif()
endif()

# Custom property to disable console window on Windows
set_property(TARGET ${PROJECT_NAME} PROPERTY WIN32_EXECUTABLE true)

#Qt
target_link_libraries(PLS_Analysis_GUI PRIVATE Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::PrintSupport)

#Armadillo
target_link_libraries(${PROJECT_NAME} PRIVATE ${ARMADILLO_LIBRARIES})

#MLPACK
target_link_libraries(${PROJECT_NAME} PRIVATE mlpack::mlpack)


