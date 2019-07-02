#!/bin/bash

set -e

mkdir linux_dependencies
cd linux_dependencies

# Install packages
apt-get update
apt-get install -y wget gnupg vim
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list
apt-get update
apt-get install -y build-essential libqwt-dev qt5-default libqglviewer-dev-qt5 subversion git cmake intel-mkl-64bit-2019.4-070 ocl-icd-libopencl1 opencl-headers ocl-icd-opencl-dev


# Intel OpenCL
mkdir IntelOpenCl
cd IntelOpenCl
wget https://github.com/intel/compute-runtime/releases/download/19.20.13008/intel-gmmlib_19.1.1_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.20.13008/intel-igc-core_1.0.4-1880_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.20.13008/intel-igc-opencl_1.0.4-1880_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.20.13008/intel-opencl_19.20.13008_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.20.13008/intel-ocloc_19.20.13008_amd64.deb
dpkg -i *.deb
cd ..


# libGLViewer
wget http://www.libqglviewer.com/src/libQGLViewer-2.7.1.tar.gz
tar -xzf libQGLViewer-2.7.1.tar.gz
cd libQGLViewer-2.7.1/QGLViewer
qmake
make -j8
cp libQGLViewer-qt5.so /usr/lib/libQGLViewer.so
cd ../..

# Chrono
mkdir chrono
cd chrono
git clone https://github.com/projectchrono/chrono.git .
git checkout ab6a969e4035a67c835026b3e3 
mkdir build
cd build
cmake -DENABLE_MODULE_MKL=ON ..
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=INTEL
make -j8
make install
cd ../..

# CLBlast
git clone https://github.com/CNugteren/CLBlast.git cblast
cd cblast
git checkout 0c9411c84465d14d2de330
mkdir build
cd build
cmake ..
make -j8
make install
cd ../..

cd ..
rm -rf include_mkl include_chrono
ln -s /opt/intel/mkl/include include_mkl
ln -s linux_dependencies/chrono/ include_chrono