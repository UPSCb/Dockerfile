#!/bin/bash
mkdir index
for f in *.fasta; do 
 indexdb --ref $f,./index/${f/.fasta/}
done 
