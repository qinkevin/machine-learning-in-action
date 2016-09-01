#-*- coding: UTF-8 -*-

from numpy import *
import operator


def createDataSet():
	group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels=['A','A','B','B']
	return group,labels

def classify0(inX,dataSet,labels,k):
	dataSetSize=dataSet.shape[0] #dataSet的长度
	diffMat=tile(inX,(dataSetSize,1))-dataSet#差
	sqDiffMat=diffMat**2#各自求平方
	sqDistances=sqDiffMat.sum(axis=1)#求和
	distances=sqDistances**0.5
	sortedDistIndicies=distances.argsort()#从小到大的排序的index
	classCount={}
	for i in range(k):
		voteIlabel=labels[sortedDistIndicies[i]] #求label
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #转化为计数字典
	sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)#对计数字典进行排序
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr=open(filename)
	numberOfLines=len(fr.readlines())
	returnMat=zeros((numberOfLines,3))
	classLabelVector=[]
	fr=open(filename)
	index=0
	for line in fr.readlines():
		line=line.strip()
		listFromLine=line.split('\t')
		returnMat[index,:]=listFromLine[0:3]
		classLabelVector.append(listFromLine[-1])
		index+=1
	return returnMat,classLabelVector

datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
