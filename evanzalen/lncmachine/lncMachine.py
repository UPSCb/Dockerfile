#!/usr/bin/python
# Copyright(c) 2018 - Halise Busra Cagirici <busraangin@gmail.com>, Hikmet Budak <hikmet.budak@icloud.com>
# status: production


### This script can be run 
###
###


from Bio import SeqIO
from Bio import Seq
from Bio.SeqUtils import ProtParam
from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA
import sys
import os
import re
import time
from optparse import OptionParser,OptionGroup

import numpy as np
import pandas as pd
from pandas import read_csv
import math

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score

import pickle


def main():
	usage = "\n%prog  [options]\n\nlncMachine requires python3 or newer.\n\nTo build a Random Forest prediction model from a coding and a noncoding data (FASTA):\npython3 lncMachine.py -c coding.fasta -n noncoding.fasta --train \n\nTo build prediction models using nine machine learning algorithms:\npython3 lncMachine.py -c coding.fasta -n noncoding.fasta --train --all\n\nTo build a Random Forest prediction model from a CSV file containing at least two classes:\npython3 lncMachine.py -i features.csv --train\n\nTo predict coding probability from a FASTA file:\npython3 lncMachine.py -c coding.fasta --model prebuiltin_model.sav"
	parser = OptionParser(usage,version="%prog  0.1")
	parser.add_option("-c","--cod",action="store",dest="coding_file",help="Coding sequences in fasta format")
	parser.add_option("-n","--noncod",action="store",dest="noncoding_file",help="Noncoding sequences in fasta format. Required for training.")
	parser.add_option("--train",action="store_true", dest="train",help="Train using coding and noncoding datasets [default:RandomForestClassifier(random_state=1, n_jobs=-1)]. Both -n and -c required.")
	parser.add_option("--all",action="store_true", dest="all9",help="Build models for all nine algorithms.")	
#	parser.add_option("--model",action="store", dest="prediction_model",help="Prediction model in .sav format [optional]", default=LogisticRegression(random_state=1, n_jobs=-1))
	parser.add_option("--model",action="store", dest="prediction_model",help="Prediction model in .sav format [optional]")
	parser.add_option("--algorithm",action="store", dest="algorithm",help="Use specified machine learning prediction algorihm i.e. RandomForestClassifier(random_state=1, n_jobs=-1)")
	parser.add_option("-i",action="store", dest="icsv",help="Tab separated CSV file containing class and feature information [optional]")
	parser.add_option("-o", "--out",action="store", dest="output_file",help="Output file name for classification (1 for coding and 0 for noncoding) and the features [default:'features.csv'].", default="features.csv")
	(options,args)=parser.parse_args()

	if options.icsv:
		dataFrame=read_csv(options.icsv,sep="\t")
		dataFrame=dataFrame[["readID", "class", "len", "orflen", "pI", "GC%"]]
		if not options.output_file: options.output_file = options.icsv+".features.csv"

	if options.train:
	
		if not options.icsv:
			print("Feature set is not provided. Features will be extracted from fasta files.")
			if not options.coding_file and not options.noncoding_file:
				parser.print_help()
				sys.exit(0)
		
			data_dict=initialize_dataFrame()
			print("data_dict has been initialized!")
			print(data_dict.keys())
			data_dict=print_features(options.coding_file, data_dict, "coding")
			data_dict=print_features(options.noncoding_file, data_dict, "noncoding")
			dataFrame=pd.DataFrame(data=data_dict)
		
			dataFrame.to_csv(options.output_file, sep='\t',index=False)

		if options.all9: build_prediction_model_all(dataFrame, options.output_file)
		elif options.algorithm: build_prediction_model_by_algorithm(dataFrame, options.algorithm, options.output_file)
		else: build_prediction_model(dataFrame, options.output_file)
	
	else:
		if not options.icsv:
			if not options.coding_file:
				parser.print_help()
				sys.exit(0)

			data_dict=initialize_dataFrame()
			data_dict=print_features(options.coding_file, data_dict, "coding")
			dataFrame=pd.DataFrame(data=data_dict)

		if options.prediction_model: coding_prediction(dataFrame, options.prediction_model, options.output_file)
		else: coding_prediction_all(dataFrame)


def build_prediction_model(dataFrame,output_file):
#	selected=[col for col in dataFrame.columns if (("class" not in col) and ("readID" not in col))]
	selected=[col for col in dataFrame.columns if (("class" not in col) and ("readID" not in col) and ("proba" not in col))]
	X = data[selected]
	Y = data["class"]
	
	prediction_model=RandomForestClassifier(random_state=1, n_jobs=-1)

	scores = cross_val_score(prediction_model, X, Y, cv=10)
	print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (prediction_model, 100*scores.mean(), 100*scores.std() * 2))

	clf=prediction_model.fit(X,Y)
	np.set_printoptions(precision=2)
	fname = '%s.%s_model.sav' % (output_file, prediction_model)
	pickle.dump(clf, open(fname, 'wb'))

def coding_prediction(dataFrame, prediction_model, output_file):
#	selected=[col for col in dataFrame.columns if (("class" not in col) and ("readID" not in col))]
	selected=[col for col in dataFrame.columns if (("class" not in col) and ("readID" not in col) and ("proba" not in col))]
	X_test = dataFrame[selected]
	y_test = dataFrame["class"]

	clf = pickle.load(open(prediction_model, 'rb'))
	y_pred=clf.predict(X_test)
	
	print("%20s | Accuracy: %0.2f%% |  " % (prediction_model, 100*clf.score(X_test,y_test)))
	print(confusion_matrix(y_test, y_pred))
	print(classification_report(y_test,y_pred))
#	print(roc_auc_score(y_test,y_pred))
	dataFrame["class"]=y_pred
	y_proba=clf.predict_proba(X_test)[:, 1]
	dataFrame["proba"]=y_proba
	dataFrame.to_csv(output_file,sep='\t',index=False)
    
def build_prediction_model_all(dataFrame, output_file):
#	selected=[col for col in dataFrame.columns if (("class" not in col) and ("readID" not in col))]
	selected=[col for col in dataFrame.columns if (("class" not in col) and ("readID" not in col) and ("proba" not in col))]
	X = dataFrame[selected]
	Y = dataFrame["class"]

	names = ["QDA", "NearestNeighbors", "DecisionTree", "RandomForest", "NeuralNet", "AdaBoost",
			 "NaiveBayes", "LogisticRegression", "LinearSVM"
			]

	classifiers = [
		QuadraticDiscriminantAnalysis(),
		KNeighborsClassifier(n_jobs=-1),
		DecisionTreeClassifier(random_state=1),
		RandomForestClassifier(random_state=1, n_jobs=-1),
		MLPClassifier(random_state=1),
		AdaBoostClassifier(random_state=1),
		GaussianNB(),
		LogisticRegression(random_state=1, n_jobs=-1),
		SVC(kernel="linear", random_state=1)
	]
	for name, clf in zip(names, classifiers):
		scores = cross_val_score(clf, X, Y, cv=10)
		print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100*scores.mean(), 100*scores.std() * 2))

		clf=clf.fit(X,Y)
		np.set_printoptions(precision=2)
		fname = output_file+'.'+name+'_model.sav'
		pickle.dump(clf, open(fname, 'wb'))

def build_prediction_model_by_algorithm(dataFrame, clf, output_file):
#	selected=[col for col in dataFrame.columns if (("class" not in col) and ("readID" not in col))]
	selected=[col for col in dataFrame.columns if (("class" not in col) and ("readID" not in col) and ("proba" not in col))]
	X = dataFrame[selected]
	Y = dataFrame["class"]

	scores = cross_val_score(clf, X, Y, cv=10)
	print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % ("your_prediction_model", 100*scores.mean(), 100*scores.std() * 2))

	clf=clf.fit(X,Y)
	np.set_printoptions(precision=2)
	fname = output_file+'.prediction_model.sav'
	pickle.dump(clf, open(fname, 'wb'))



def coding_prediction_all(dataFrame):
#	selected=[col for col in dataFrame.columns if (("class" not in col) and ("readID" not in col))]
	selected=[col for col in dataFrame.columns if (("class" not in col) and ("readID" not in col) and ("proba" not in col))]
	X_test = dataFrame[selected]
	y_test = dataFrame["class"]

	names = ["QDA", "NearestNeighbors", "DecisionTree", "RandomForest", "NeuralNet", "AdaBoost",
			 "NaiveBayes", "LogisticRegression", "LinearSVM"
			]

	classifiers = [
		QuadraticDiscriminantAnalysis(),
		KNeighborsClassifier(n_jobs=-1),
		DecisionTreeClassifier(random_state=1),
		RandomForestClassifier(random_state=1, n_jobs=-1),
		MLPClassifier(random_state=1),
		AdaBoostClassifier(random_state=1),
		GaussianNB(),
		LogisticRegression(random_state=1, n_jobs=-1),
		SVC(kernel="linear", random_state=1)
	]

	for name, clf in zip(names, classifiers):
		clf = pickle.load(open(name+'_model.sav', 'rb'))
		y_pred=clf.predict(X_test)
# 		print("%20s | Accuracy: %0.2f%% |  " % (name, 100*clf.score(X_test,y_test)))
# 		print(confusion_matrix(y_test, y_pred))
# 		print(classification_report(y_test,y_pred))
#		print(roc_auc_score(y_test,y_pred))
		dataFrame["class"]=y_pred
		y_proba=clf.predict_proba(X_test)[:, 1]
		dataFrame["proba"]=y_proba
		
		dataFrame.to_csv(name+'_out.csv',sep='\t',index=False)



def initialize_dataFrame():
#	header=["readID", "class", "len", "orflen", "orfcov", "pI", "ORF_integrity", "fickett", "fickett_cds", "Hexamer", "GC%", "A", "T", "G", "C", "AAA", "AAT", "AAG", "AAC", "ATA", "ATT", "ATG", "ATC", "AGA", "AGT", "AGG", "AGC", "ACA", "ACT", "ACG", "ACC", "TAA", "TAT", "TAG", "TAC", "TTA", "TTT", "TTG", "TTC", "TGA", "TGT", "TGG", "TGC", "TCA", "TCT", "TCG", "TCC", "GAA", "GAT", "GAG", "GAC", "GTA", "GTT", "GTG", "GTC", "GGA", "GGT", "GGG", "GGC", "GCA", "GCT", "GCG", "GCC", "CAA", "CAT", "CAG", "CAC", "CTA", "CTT", "CTG", "CTC", "CGA", "CGT", "CGG", "CGC", "CCA", "CCT", "CCG", "CCC", "AA", "AT", "AG", "AC", "TA", "TT", "TG", "TC", "GA", "GT", "GG", "GC", "CA", "CT", "CG", "CC"]
	header=["readID", "class", "len", "orflen", "pI", "GC%", "proba"]
	data_dict={}
	for h in header: 
#		if h != "readID": data_dict[h]=[]
		data_dict[h]=[]

	return data_dict

def print_features(fasta_file, data_dict, annot):
	if annot=="coding": annot=1
	elif annot=="noncoding": annot=0

	for seq in SeqIO.parse(fasta_file,"fasta"):
	  seqid = seq.id
	  seqDNA=seq.seq
	  seqDNA=seqDNA.upper()
	  seqlen=len(seqDNA)
	  seqCDS,orf_integrity = FindCDS(seqDNA).longest_orf()
	#  seqProt=PA(str(Seq(seqCDS).translate().strip("*")))
	  Prot=PA(str(seqCDS.translate().strip("*")))
	  seqProt=Prot.sequence
	  orflen=len(seqProt)
	  if len(seqProt)> 0: isoelectric_point = Prot.isoelectric_point()
	  else: isoelectric_point = 0.0	  
	  gc=(seqDNA.count("G")+seqDNA.count("C"))*100.0/len(seqDNA)
	  
	  data_dict["readID"].append(seqid)
	  data_dict["class"].append(annot)
	  data_dict["len"].append(seqlen)
	  data_dict["orflen"].append(orflen)
	  data_dict["pI"].append(isoelectric_point)
	  data_dict["GC%"].append(gc)
	  
	return data_dict


class FindCDS:
	'''
	Find the most like CDS in a given sequence 
	The most like CDS is the longest ORF found in the sequence
	When having same length, the upstream ORF is printed
	modified from source code of CPAT 1.2.1 downloaded from https://sourceforge.net/projects/rna-cpat/files/?source=navbar
	'''
	def __init__(self,seq):
		self.seq = seq
		self.result = (0,0,0,0)
		self.longest = 0
		self.basepair = {"A":"T","T":"A","U":"A","C":"G","G":"C","N":"N","X":"X"}

	def _reversecompliment(self):
		return "".join(self.basepair[base] for base in self.seq)[::-1]

	def get_codons(self,frame_number):
		'''
		Record every nucleotide triplet and its coordinate position for input sequence in one frame
		'''
		coordinate = frame_number
		while coordinate + 3 <= len(self.seq):
			yield (self.seq[coordinate:coordinate+3], coordinate)
			coordinate += 3 
	
	def find_longest_in_one(self,myframe,start_codon,stop_codon):
		'''
		find the longest ORF in one reading myframe
		'''
		triplet_got = self.get_codons(myframe)	
		starts = start_codon
		stops = stop_codon
		'''
		Extend sequence by triplet after start codon encountered
		End ORF extension when stop codon encountered
		'''
		while True:
			try: 
				codon,index = triplet_got.__next__()
			except StopIteration:
				break 
			if codon in starts and codon not in stops:
				'''
				find the ORF start
				'''
				orf_start = index
				end_extension = False
				while True:
					try: 
						codon,index = triplet_got.__next__()
					except StopIteration:
						end_extension = True
						integrity = -1
					if codon in stops:
						integrity = 1
						end_extension = True
					if end_extension:
						orf_end = index + 3
						Length = (orf_end - orf_start)
						if Length > self.longest:
							self.longest = Length
							self.result = [orf_start,orf_end,Length,integrity]
						if Length == self.longest and orf_start < self.result[0]:
							'''
							if ORFs have same length, return the one that if upstream
							'''
							self.result = [orf_start,orf_end,Length,integrity]
						break

	def longest_orf(self,start_codon={"ATG":None}, stop_codon={"TAG":None,"TAA":None,"TGA":None}):
		return_orf = ""
		for frame in range(3):
			self.find_longest_in_one(frame,start_codon,stop_codon)
		return_orf = self.seq[self.result[0]:self.result[1]][:]
		start_coordinate = self.result[0]
		orf_integrity = self.result[3]
		return return_orf,orf_integrity

if __name__ == '__main__':
	main()
