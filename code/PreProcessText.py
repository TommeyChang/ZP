#!/usr/bin/python
#coding=utf-8

import os, sys, logging
import re, random


class FilePreprocess:
    def __init__(self, datasetPath):
        self.datasetPath =  datasetPath 
        self.logger = logging.getLogger("FilePreprocess")
        self.logger.setLevel(logging.INFO)
        logFormat = logging.Formatter('%(asctime)s :%(name)s :%(levelname)s :%(message)s')
        teminalLog = logging.StreamHandler()
        teminalLog.setFormatter(logFormat)
        teminalLog.setLevel(logging.INFO)
        self.logger.addHandler(teminalLog)

    def getAllCorefFiles(self):
        self.corefFiles = []
        self.logger.info("Start getting the coref file names...")
        for fpath, dirs, fileNames in os.walk(self.datasetPath):
            for perFile in fileNames:
                if 'coref' in perFile:
                    self.corefFiles.append(os.path.join(fpath, perFile))
        self.logger.info("Finish getting the coref file names")
    
    def sortAndMerge(self, cleanRate = 6):
        fpOutFile = open(self.datasetPath + '/ProcessedText.txt','w')
        #fpHalfOutFile = open(self.datasetPath + '/HalfProcessText.txt','w')
        delPatter = re.compile(r'<(.*?)>|(\*OP\*\s|(\*T\*(-(\d)*)*\s|(\*RNR\*(-(\d)*)*\s|\*-1)))')
        blankPatter = re.compile(r'\s\s')
        annotePatter = re.compile(r'\*[Pp][Rr][Oo]\*(\s)')
        self.logger.info("Start processing the coref files...")
        lineCountTotal = 0
        lineCountExist = 0
        timeCountExist = 0
        lineCountNone = 0
        for perCoreFileName in self.corefFiles:            
            fpPerFile = open(perCoreFileName)
            for line in fpPerFile:
                delTagLine = delPatter.sub('', line)
                delLine = blankPatter.sub(r' ', delTagLine)
                if len(delLine) > 1:                         
                    lineCountTotal += 1
                    proMatches = annotePatter.finditer(delLine)    #generate a iterate of the *pro* matches                   
                    preProTail = 0 # record last pro position
                    tokenCounts = [] # store every pos of the pro
                    for eachMatch in proMatches:                                           
                        proSpan = eachMatch.span()   # get the span of the *pro* or *PRO*
                        #find another position of 
                        tokenCount = []
                        for letter in delLine[ preProTail : ( proSpan[0] ) ]:                            
                            if letter == ' ':
                                tokenCount.append('0')                                
                        tokenCounts += tokenCount
                        tokenCounts.append('1')
                        timeCountExist += 1
                        preProTail = proSpan[1] 
                    else:
                        if len(tokenCounts) > 0 or random.randint(0,9) > cleanRate:
                            # Only preserve these lines that contain pro, and random select some no pro lines
                            tokenCount = []
                            for letter in delLine[ preProTail : ]:                            
                                if letter == ' ':
                                    tokenCount.append('0')                                
                            tokenCounts += tokenCount                        
                    
                            result = annotePatter.sub('', delLine)                    
                            fpOutFile.write(result[:-1] + '\t' + ','.join(map(str,tokenCounts)) + '\n')
                            lineCountExist += 1
                    
            fpPerFile.close()
            if lineCountTotal % 100 == 0:
                self.logger.info("Finish processing %d lines, %d lines exist pro, %d times." % (lineCountTotal, lineCountExist, timeCountExist))
        fpOutFile.close()
        self.logger.info("Finish processing %d lines, %d lines exist pro, %d times." % (lineCountTotal, lineCountExist, timeCountExist))
        #fpHalfOutFile.close()
        


if __name__ == '__main__':
    parameterNumber = len(sys.argv)
    if parameterNumber < 2:
        exit(-1)    
    datasetPath = sys.argv[1]
    processer = FilePreprocess(datasetPath)
    processer.getAllCorefFiles()
    if parameterNumber < 3:
        processer.sortAndMerge()
    else:
        processer.sortAndMerge(sys.argv[2])