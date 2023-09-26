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

function DIGINETICA () {
    if [ ! -e "DIGINETICA/dataset-train-diginetica.zip" ]; then
        mkdir -p DIGINETICA
        echo "Please download dataset-train-diginetica.zip to `pwd`/DIGINETICA and unzip it."
        echo "The zip file is available at https://drive.google.com/drive/folders/0B7XZSACQf0KdXzZFS21DblRxQ3c?resourcekey=0-3k4O5YlwnZf0cNeTZ5Y_Uw&usp=sharing"
    fi
}

function AMAZON_M2 () {
    if [ ! -e "AMAZON_M2/sessions_train.csv" ]; then
        mkdir -p AMAZON_M2
        echo "Please download sessions_train.csv to `pwd`/AMAZON_M2 and unzip it."
        echo "The zip file is available at https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge"
    fi
}

function Epinions () {
    if [ ! -e "Epinions/soc-Epinions1.txt" ]; then
        mkdir -p Epinions
        wget -P Epinions https://snap.stanford.edu/data/soc-Epinions1.txt.gz
        gunzip Epinions/soc-Epinions1.txt.gz
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

function LFM2B_1MON () {
    if [ ! -e "LFM2B_1MON/listening_events.tsv" ]; then
        mkdir -p LFM2B_1MON
        wget -P LFM2B_1MON http://www.cp.jku.at/datasets/LFM-2b/recsys22/listening_events.tsv.bz2
        bzip2 -d LFM2B_1MON/listening_events.tsv.bz2
    fi
}

function Twitch100K () {
    if [ ! -e "Twitch100K/100k_a.csv" ]; then
        mkdir -p Twitch100K
        wget -P Twitch100K https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/twitch/100k_a.csv
    fi
}

function LastFM_TAG () {
    if [ ! -e "LastFM_TAG/user_taggedartists-timestamps.dat" ]; then
        mkdir -p LastFM_TAG
        wget https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
        unzip -d LastFM_TAG hetrec2011-lastfm-2k.zip
        rm hetrec2011-lastfm-2k.zip
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
elif [ $1 = "DIGINETICA" ]; then
    DIGINETICA
elif [ $1 = "AMAZON_M2" ]; then
    AMAZON_M2
elif [ $1 = "Epinions" ]; then
    Epinions
elif [ $1 = "Ciao" ]; then
    Ciao
elif [ $1 = "Ciao_PART" ]; then
    Ciao_PART
elif [ $1 = "LFM2B_1MON" ]; then
    LFM2B_1MON
elif [ $1 = "Twitch100K" ]; then
    Twitch100K
elif [ $1 = "LastFM_TAG" ]; then
    LastFM_TAG
# elif [ $1 = "ALL" ]; then
#     ML100K
#     ML1M
#     ML10M
#     ML25M
#     # BookCrossing
#     # Amazon
#     DIGINETICA
#     AMAZON_M2
else
    echo "Invalid Argument"
fi