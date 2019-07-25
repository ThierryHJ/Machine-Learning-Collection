import csv
import math
import sys


def parseData(data):
    with open(data, 'r', encoding='utf-8') as csv_file:
        lines = csv.reader(csv_file, delimiter=',')
        attrList = []
        classList = []
        trainList = []
        labelList = []
        treeBoard = []
        for index, i in enumerate(lines):
            if index == 0:
                attrList.append(i[:-1])
                classList.append(i[-1])
            if index != 0:
                treeBoard.append(i)
                trainList.append(i[:-1])
                labelList.append(i[-1])

    return treeBoard,  attrList


def makeAlias(treeBoard):
    #Convert alias
    aliasDic ={}
    aliasDic['democrat'] = "+"
    aliasDic['republican'] = '-'
    aliasDic['A'] = '+'
    aliasDic['notA'] = '-'
    aliasDic['yes'] = '+'
    aliasDic['no'] = '-'

    # Transforming value in treeBoard
    boardLen = len(treeBoard)
    boardWid = len(treeBoard[0])

    for row in range(boardLen):
        for col in range(boardWid):
            if col == boardWid - 1:
                treeBoard[row][col] = aliasDic[treeBoard[row][col]]

    return treeBoard



def calEntropy(tree):

    if tree == None:
        entropy = 0
    else:
        labelList = [i[-1] for i in tree]
        total = len(tree)
        positiveCount = labelList.count('+')
        negativeCount = labelList.count('-')

        if positiveCount == 0 or negativeCount == 0:
            entropy = 0

        else:
            entropy = positiveCount / total * math.log2(total / positiveCount) + \
                      negativeCount / total * math.log2(total / negativeCount)
            errorRate = min(positiveCount, negativeCount) / total

    return entropy


def splitTree(attr, treeBoard):
    tree1 = []
    tree2 = []
    firstValue = None
    for index, row in enumerate(treeBoard):
        if index == 0:
            firstValue = row[attr]
        if row[attr] == firstValue:
            tree1.append(row)
        else:
            tree2.append(row)

    return tree1, tree2


def DecisionTreeRoot(tree, attrList, bestAttr):
    #     labelList = [i[-1] for i in treeBoard]
    trainList  = [i[:-1] for i in tree]
    # Helper function calEntropy
    rootEntropy = calEntropy(tree)

    # Search for best candidate
    bestMutualInfo1 = 0
    bestTree1 = None
    bestTree2 = None
    for attr in range(len(trainList[0])):
        if attr != bestAttr:

            # Helper function splitTree
            tree1, tree2 = splitTree(attr, tree)
            firstChildEntropy = (len(tree1) / len(tree)) * calEntropy(tree1) + \
                                (len(tree2) / len(tree)) * calEntropy(tree2)

            mutualInfo1 = rootEntropy - firstChildEntropy
            if mutualInfo1 >= 0.1 and mutualInfo1 > bestMutualInfo1:
                bestMutualInfo1 = mutualInfo1
                bestAttr = attr
                bestTree1 = tree1
                bestTree2 = tree2

    bestTree1Entropy = calEntropy(bestTree1)
    bestTree2Entropy = calEntropy(bestTree2)


    return bestTree1, bestTree1Entropy, bestTree2, bestTree2Entropy,bestAttr


def DecisionTreeChild(parentTree, attrList, bestAttr):
    vote1 = None
    vote2 = None
    model1 = None
    model2 = None

    bestTree3, bestTree3Entropy, bestTree4, \
    bestTree4Entropy, bestAttr = DecisionTreeRoot(parentTree, attrList, bestAttr)

    if bestTree3 == None and bestTree4 == None:
        parentTreeLabel = [i[-1] for i in parentTree]
        minor = min(parentTreeLabel.count('+'),parentTreeLabel.count('-'))


    else:
        bestTree3Label = [i[-1] for i in bestTree3]
        print('|',printTree(attrList, bestAttr, bestTree3, bestTree3Label))
        model1 = buildModel(attrList,bestAttr,bestTree3,bestTree3Label)
        minor1 = min(bestTree3Label.count('+'),bestTree3Label.count('-'))


        bestTree4Label = [i[-1] for i in bestTree4]
        print('|',printTree(attrList, bestAttr, bestTree4, bestTree4Label))
        model2 = buildModel(attrList, bestAttr, bestTree4, bestTree4Label)
        minor2 = min(bestTree4Label.count('+'), bestTree4Label.count('-'))


        minor = minor1 + minor2


    return minor,bestAttr,model1,model2



def printTree(attrList,bestAttr,bestTree,bestTreeLabel):
    return  "%s = %s: [%d+/%d-]" %  (attrList[0][bestAttr],
                                     bestTree[0][bestAttr],
                                     bestTreeLabel.count('+'),
                                     bestTreeLabel.count('-'))

#
# def testData(bestAttr,bestAttrChild1,bestAttrChild2,vote1,vote3,vote2=None,vote4=None):
#     if bestAttr == bestAttrChild1:
#

def buildModel(attrList,bestAttr,bestTree,bestTreeLabel):
    if bestTreeLabel.count('+') == max(bestTreeLabel.count('+'), bestTreeLabel.count('-')):
        vote = '+'
    else:
        vote = '-'
    model = attrList[0][bestAttr],bestTree[0][bestAttr],vote

    return model


def buildFullModel(model1, model3, model4, model2, model5, model6):
    list1 = [model1, model3, model4]
    list2 = [model2, model5, model6]
    listFullPre = [list1, list2]

    listFull = []
    ModelDic = {}
    for i in listFullPre:
        if i[1] == None and i[2] == None:
            i = i[0]
            listFull.append({i[0]: i[1], 'vote': i[-1]})

        else:
            listFull.append({i[0][0]: i[0][1], i[1][0]: i[1][1], 'vote': i[1][-1]})
            listFull.append({i[0][0]: i[0][1], i[2][0]: i[2][1], 'vote': i[2][-1]})

    return listFull


def testingData(testData, attrListTest, modelList):
    rowCount = 0
    count = 0

    for dic in modelList:
        aList = [i for i in dic.keys() if i != 'vote']
        if len(aList) == 2:
            attr1 = aList[0]
            attr2 = aList[1]
            indexDic = {}
            for index, attr in enumerate(attrListTest[0]):
                if attr == attr1:
                    indexDic[attr1] = index
                elif attr == attr2:
                    indexDic[attr2] = index
            #             print(indexDic)
            for row in testData:
                if row[indexDic[attr1]] == dic[attr1] and row[indexDic[attr2]] == dic[attr2]:
                    if row[-1] != dic['vote']:
                        count += 1
                    rowCount += 1
        elif len(aList) == 1:
            attr3 = aList[0]
            indexDic = {}
            for index, attr in enumerate(attrListTest[0]):
                if attr == attr3:
                    indexDic[attr3] = index

            for row in testData:
                if row[indexDic[attr3]] == dic[attr3]:
                    if row[-1] != dic['vote']:
                        count += 1
                    rowCount += 1

    errorRate = count / len(testData)

    return errorRate


def main():
    trainData = sys.argv[1]
    testData = sys.argv[2]
    # Read Data based on input
    treeBoard,attrList = parseData(trainData)
    treeBoardTest, attrListTest = parseData(testData)

    # Make alias for label
    tree = makeAlias(treeBoard)
    treeTest = makeAlias(treeBoardTest)

    #print root
    labelList = [i[-1] for i in tree]
    print("[%d+/%d-]" % (labelList.count('+'),
                         labelList.count('-')))

    # Learn the first layer tree
    bestAttr = None
    model1,model2,model3,model4,model5,model6= (None, )*6
    bestTree1, bestTree1Entropy, \
    bestTree2, bestTree2Entropy, bestAttr = DecisionTreeRoot(treeBoard, attrList, bestAttr)

    # Left
    if bestTree1 != None:
        bestTree1Label = [i[-1] for i in bestTree1]
        print(printTree(attrList, bestAttr, bestTree1, bestTree1Label))
        model1 = buildModel(attrList,bestAttr,bestTree1,bestTree1Label)

        # Learn the second layer tree
        if bestTree1Entropy != 0:
            minor1,bestAttrChild1,model3,model4 = DecisionTreeChild(bestTree1, attrList, bestAttr)
        else:
            minor1 = 0

    # Right
    if bestTree2 != None:
        bestTree2Label = [i[-1] for i in bestTree2]
        print(printTree(attrList, bestAttr, bestTree2, bestTree2Label))
        model2 = buildModel(attrList,bestAttr,bestTree2,bestTree2Label)

        # Learn the second layer tree
        if bestTree2Entropy != 0:
            minor2,bestAttrChild2,model5,model6 = DecisionTreeChild(bestTree2, attrList, bestAttr)
        else:
            minor2 = 0


    errorRateTrain = (minor1+minor2) / len(tree)

    # Test data
    modelList = buildFullModel(model1, model3, model4, model2, model5, model6)
    errorRateTest = testingData(treeTest, attrListTest, modelList)
    errorRateTrain2 = testingData(tree, attrListTest, modelList)

    print('error(train):',errorRateTrain)
    print('error(test):', errorRateTest)



if __name__ == "__main__":
    main()

