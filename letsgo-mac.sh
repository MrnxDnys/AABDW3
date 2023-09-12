#!/usr/bin/env sh

echo I assume you have Python and Jupyter installed and accessible in this environment

CURRENT_HOME="$( cd "$(dirname "$0")" ; pwd -P )"
export SPARK_HOME=$CURRENT_HOME/spark-3.2.1-bin-hadoop2.7
export PYSPARK=$SPARK_HOME/bin/pyspark

export PYSPARK_DRIVER_PYTHON="jupyter" 
export PYSPARK_DRIVER_PYTHON_OPTS="notebook" 
export PYSPARK_PYTHON=/Users/marnix/opt/anaconda3/envs/AABDW3/bin/python

chmod +x $JAVA_HOME/bin/*
chmod +x $SPARK_HOME/bin/*

echo Ready to start
echo $PYSPARK

$PYSPARK
