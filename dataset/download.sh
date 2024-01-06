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

function DIGINETICA () {
    if [ ! -e "DIGINETICA/dataset-train-diginetica.zip" ]; then
        mkdir -p DIGINETICA
        echo "Please download dataset-train-diginetica.zip to `pwd`/DIGINETICA and unzip it."
        echo "The zip file is available at https://drive.google.com/drive/folders/0B7XZSACQf0KdXzZFS21DblRxQ3c?resourcekey=0-3k4O5YlwnZf0cNeTZ5Y_Uw&usp=sharing"
    fi
}

function Ciao () {
    if [ ! -e "Ciao/ciao_with_rating_timestamp/rating_with_timestamp.mat" ]; then
        wget https://www.cse.msu.edu/~tangjili/datasetcode/ciao_with_rating_timestamp.zip
        unzip -d Ciao ciao_with_rating_timestamp.zip
        rm ciao_with_rating_timestamp.zip
    fi
}

function Ciao_PART () {
    if [ ! -e "Ciao_PART/ciao_with_rating_timestamp_txt/rating_with_timestamp.txt" ]; then
        wget https://www.cse.msu.edu/~tangjili/datasetcode/ciao_with_rating_timestamp_txt.zip
        unzip -d Ciao_PART ciao_with_rating_timestamp_txt.zip
        rm ciao_with_rating_timestamp_txt.zip
    fi
}

function Twitch100K () {
    if [ ! -e "Twitch100K/100k_a.csv" ]; then
        mkdir -p Twitch100K
        wget -P Twitch100K https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/twitch/100k_a.csv
    fi
}

function Yelp () {
    if [ ! -e "Yelp/yelp_training_set" ]; then
        mkdir -p Yelp
        echo "Please download yelp_training_set.zip to `pwd`/Yelp and unzip it."
        echo "The zip file is available at https://www.kaggle.com/competitions/yelp-recsys-2013/data"
    fi
}

function FourSquare () {
    if [ ! -e "FourSquare/dataset_TSMC2014_TKY.csv" ]; then
        mkdir -p FourSquare
        echo "Please download archive.zip to `pwd`/FourSquare and unzip it."
        echo "The zip file is available at https://www.kaggle.com/datasets/chetanism/foursquare-nyc-and-tokyo-checkin-dataset"
    fi
}

if [ $1 = "INIT" ]; then
    ls | grep -v -E 'download.sh' | xargs rm -r
elif [ $1 = "ML100K" ]; then
    ML100K
elif [ $1 = "ML1M" ]; then
    ML1M
elif [ $1 = "DIGINETICA" ]; then
    DIGINETICA
elif [ $1 = "Ciao" ]; then
    Ciao
elif [ $1 = "Ciao_PART" ]; then
    Ciao_PART
elif [ $1 = "Twitch100K" ]; then
    Twitch100K
elif [ $1 = "Yelp" ]; then
    Yelp
elif [ $1 = "FourSquare" ]; then
    FourSquare
else
    echo "Invalid Argument"
fi