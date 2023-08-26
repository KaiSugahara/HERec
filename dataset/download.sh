#!/bin/bash

function ML100K () {
    if [ ! -e "ML100K" ]; then
        wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
        unzip -d ML100K ml-100k.zip
        rm ml-100k.zip
    fi
}

function ML1M () {
    if [ ! -e "ML1M" ]; then
        wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
        unzip -d ML1M ml-1m.zip
        rm ml-1m.zip
    fi
}

function ML10M () {
    if [ ! -e "ML10M" ]; then
        wget https://files.grouplens.org/datasets/movielens/ml-10m.zip
        unzip -d ML10M ml-10m.zip
        rm ml-10m.zip
    fi
}

function ML25M () {
    if [ ! -e "ML25M" ]; then
        wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
        unzip -d ML25M ml-25m.zip
        rm ml-25m.zip
    fi
}

function BookCrossing () {
    if [ ! -e "BookCrossing" ]; then
        wget http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip
        unzip -d BookCrossing BX-CSV-Dump.zip
        rm BX-CSV-Dump.zip
    fi
}

function Amazon () {
    if [ ! -e "Amazon" ]; then
        mkdir -p Amazon
        wget -P Amazon https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/all_csv_files.csv
    fi
}

if [ $1 = "INIT" ]; then
    ls | grep -v -E 'download.sh' | xargs rm -r
elif [ $1 = "ML100K" ]; then
    ML100K
elif [ $1 = "ML1M" ]; then
    ML1M
elif [ $1 = "ML10M" ]; then
    ML10M
elif [ $1 = "ML25M" ]; then
    ML25M
# elif [ $1 = "BookCrossing" ]; then
#     BookCrossing
# elif [ $1 = "Amazon" ]; then
#     Amazon
elif [ $1 = "ALL" ]; then
    ML100K
    ML1M
    ML10M
    ML25M
    # BookCrossing
    # Amazon
else
    echo "Invalid Argument"
fi