#!/bin/bash
echo "Start"
while read line; do 
    echo "$line";
    ./squall2sparql_revised -wikidata "$line";
done < "curie_mt_400_.txt"