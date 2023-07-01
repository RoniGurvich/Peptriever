#!/bin/bash
BINDING_DATA_DIR=/data/training_sets/external/binding
mkdir -p $BINDING_DATA_DIR
echo "Downloading Pepdb"
cd $BINDING_DATA_DIR && wget http://huanglab.phys.hust.edu.cn/pepbdb/db/download/pepbdb-20200318.tgz && tar xzf pepbdb-20200318.tgz
echo "Downloading propedia"
cd $BINDING_DATA_DIR && wget http://bioinfo.dcc.ufmg.br/propedia/public/download/complex.csv -O complex.csv
echo "Downloading YAPP-CD data"
cd $BINIDNG_DATA_DIR && wget http://wnl.cs.hongik.ac.kr/yapp/download/yappcdann.csv
cd $BINIDNG_DATA_DIR && wget http://wnl.cs.hongik.ac.kr/yapp/download/protein.tar.gz
cd $BINIDNG_DATA_DIR && tar xzvf ./protein.tar.gz
cd $BINIDNG_DATA_DIR && wget http://wnl.cs.hongik.ac.kr/yapp/download/peptide.tar.gz
cd $BINIDNG_DATA_DIR && tar xzvf ./peptide.tar.gz

echo "Downloading benchmark data"
cd $BINDING_DATA_DIR && wget https://www.frontiersin.org/articles/file/downloadfile/959160_supplementary-materials_datasheets_1_csv/octet-stream/Data%20Sheet%201.CSV/1/959160 -O pos_benchmark.csv
cd $BINDING_DATA_DIR && wget https://www.frontiersin.org/articles/file/downloadfile/959160_supplementary-materials_datasheets_2_csv/octet-stream/Data%20Sheet%202.CSV/1/959160 -O neg_benchmark.csv
