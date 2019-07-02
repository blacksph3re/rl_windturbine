FROM ubuntu

RUN apt update && apt install -y subversion
RUN mkdir -p /opt/qblade
WORKDIR /opt/qblade
RUN svn checkout http://svn.code.sf.net/p/qblade/code/trunk/ .

ADD build.sh /opt/qblade/build.sh
RUN bash build.sh
ADD qblade_v09.pro /opt/qblade/qblade_v09.pro

RUN qmake && make -j8
ADD sample_projects /opt/sample_projects
ENV LD_LIBRARY_PATH=/opt/qblade/libs_unix_64bit:/opt/intel/mkl/lib/intel64:/usr/lib:/usr/local/lib

RUN apt install -y python3

ADD LoadLib.py /opt/code/LoadLib.py

CMD python3 /opt/code/LoadLib.py