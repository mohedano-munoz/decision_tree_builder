import math
import pandas as pd
import json
import uuid
import webcolors as wc

"""
Tree is represented by a dictionary.
The tree has a 'data' key a 'nodes' key and a 'clicks' key.

The 'data' key contains another dictionary with the following structure:
    'df': dataframe as str. This dataframe has:
        - no cluster_class as the last column
        - target class as the last column
        - index as the first column
        - with only selected features as columns
    'targetFeature' : name of the target feature selected by user
    'selectedFeatures': list containing the selected features by the user
    'classes' : list with unique values of the target feature
    'train_length' : length for train subset of data
    'color_scale' : list with colors for every class in classes

The 'nodes' key contains a list of dictionaries. Every element in the list is a Node 
represented by a dictionary with:
    'id': unique id of 8 hex chars
    'isRoot' : Boolean
    'fatherId' : Id of the parent node
    'classPurity' : List with the prob of each class value
    'isLeaf' : Boolean
    'class' : When isLeaf is True, the class assigned to this node
    'hasChildren' : Boolean
    'childs' :  List with the id of all the nodes that have this one in the path to Root
    'leftChild': When partition using C4.5 algorithm is done, this will be the
                 id of the node with all the observations with the chosen variable with a
                 value less than or equal to the threshold value 
    'rightChild' : When partition using C4.5 algorithm is done, this will be the
                   id of the node with all the observations with the chosen variable with a
                   value greater than the threshold value
    'pathFromRoot' : List of ordered Nodes IDs from Root till the father of this Node
    'chosenAttr' : Variable used to partition this Node using C4.5 algorithm
    'threshold' : Value of chosenAttr that maximizes Information Gain according to
                  C4.5 algorithm to partition this Node
    'entropy': Entropy value for the observations contained in this node
    'samples': Percentage of observations from original DF contained in this node

The 'clicks' key contains a dictiornary with:
    'create': number of clicks of CREATE button
    'delete': number of clicks of DELETE button
    'forceLeaf': number of clicks of FORCE LEAF button
    'resetLeaf': number of clicks of RESET LEAF button

"""

"""
C4.5 algorithm functions
Functions taken from https://github.com/barisesmer/C4.5/blob/master/c45/c45.py
"""

def splitAttribute(data, chosen_attribute, classes):
    """
    Splits the dataset based on a chosen attribute and finds the best threshold for continuous attributes.

    This function iterates through all possible thresholds for the given attribute in the dataset to determine how best to split the data into two subsets (less than or equal to the threshold and greater than the threshold). It calculates the information gain for each potential split and selects the threshold that provides the maximum gain.

    Args:
        data (pandas.DataFrame): The dataset to split, with attributes as columns.
        chosen_attribute (str): The name of the attribute to split on.
        classes (list): A list of unique class labels present in the dataset.

    Returns:
        tuple: A tuple containing:
            - best_threshold (float): The optimal threshold value for the split.
            - splitted (list of lists): A list containing two subsets of the data split by the threshold.
    """
    splitted = []
    maxEnt = -1 * float("inf")
    # None for discrete attributes, threshold value for continuous attributes
    best_threshold = None
    data_list = list(data.values)
    # sort the data according to the column.Then try all
    # possible adjacent pairs. Choose the one that
    # yields maximum gain
    indexOfAttribute = data.columns.get_loc(chosen_attribute)
    data_list.sort(key=lambda x: x[indexOfAttribute])
    for j in range(0, len(data) - 1):
        if data_list[j][indexOfAttribute] != data_list[j + 1][indexOfAttribute]:
            threshold = (data_list[j][indexOfAttribute] + data_list[j + 1][indexOfAttribute]) / 2
            less = []
            greater = []
            for row in data_list:
                if (row[indexOfAttribute] > threshold):
                    greater.append(row)
                else:
                    less.append(row)
            # Remove position 0 (index variable) from all the lists passed to gain function
            e = gain([x[1:] for x in data_list], [[x[1:] for x in less], [x[1:] for x in greater]], classes)
            if e >= maxEnt:
                splitted = [less, greater]
                maxEnt = e
                best_threshold = threshold
    return (best_threshold, splitted)


def gain(dataSet, subsets, classes):
    """
    Calculates the information gain from splitting a dataset based on a given attribute.

    This function computes the information gain by first determining the impurity of the dataset before the split and then calculating the weighted impurity after the split using disjoint subsets. The information gain is the difference between the impurity before and after the split.

    Args:
        dataSet (list of lists): The dataset where each row is a data instance.
        subsets (list of lists): A list containing disjoint subsets of the dataset created by the split.
        classes (list): A list of unique class labels present in the dataset.

    Returns:
        float: The information gain resulting from the split.
    """
    # input : data, disjoint subsets of it and attribute list
    # output : information gain
    S = len(dataSet)
    # calculate impurity before split
    impurityBeforeSplit = entropy(dataSet, classes)
    # calculate impurity after split
    weights = [len(subset) / S for subset in subsets]
    impurityAfterSplit = 0
    for i in range(len(subsets)):
        impurityAfterSplit += weights[i] * entropy(subsets[i], classes)
    # calculate total gain
    totalGain = impurityBeforeSplit - impurityAfterSplit
    return totalGain


def entropy(dataSet, classes):
    """
    Calculates the entropy of a dataset based on the distribution of classes.

    This function computes the entropy, a measure of the uncertainty or impurity in the dataset, given a list of classes. It counts the occurrences of each class in the dataset and calculates the entropy based on their probabilities.

    Args:
        dataSet (list of lists): The dataset where each row is a data instance, and the last element of each row is the class label.
        classes (list): A list of unique class labels present in the dataset.

    Returns:
        float: The calculated entropy of the dataset. Returns 0 if the dataset is empty.
    """

    S = len(dataSet)
    if S == 0:
        return 0
    num_classes = [0 for i in classes]
    for row in dataSet:
        classIndex = list(classes).index(row[-1])
        num_classes[classIndex] += 1
    num_classes = [x / S for x in num_classes]
    ent = 0
    for num in num_classes:
        ent += num * log(num)
    return abs(ent * -1)


def log(x):
    if x == 0:
        return 0
    else:
        return math.log(x, 2)

"""
Decision Tree related functions
"""

def createTree(df, targetFeature, selectedFeatures, create_btn, delete_btn, forceLeaf_btn,
               resetLeaf_btn, total_color_scale):
    """
    Function to generate a Tree.
    It initializes a tree with the data information.
    :param df:
        df is X_train_original from 'run_algorithms' function of Callback for Algorithms data generation.
        It is the X_train_not_scaled key of the PCA or any other algorithm in resulting dictionary
    :param targetFeature:
        target_feature is the name of the column from df that is the target feature for classification
    :param selectedFeatures:
        selected_features is a list with the names of the columns that are the features already selected
        by the user
    :param create_btn:
        number of clicks of CREATE button at the moment the tree is created or modified
    :param delete_btn:
        number of clicks of DELETE button at the moment the tree is created or modified
    :param forceLeaf_btn:
        number of clicks of FORCE LEAF button at the moment the tree is created or modified
    :param resetLeaf_btn:
        number of clicks of RESET LEAF button at the moment the tree is created or modified
    :param total_color_scale:
        complete list for the color scale. To be used to build the color scale list for the classes

    :return:
        This function returns a dictionary that represents a Tree without with a Root Node that has not been
        yet partitioned.
    """

    # Calculate variables to create the data part of the tree dict
    new_cols = ['index']
    for feature in selectedFeatures:
        if feature != 'index' and feature != 'cluster_class' and feature != targetFeature:
            new_cols.append(feature)
    new_cols.append(targetFeature)
    data = df[new_cols]
    # convert numpy item to python native type using val.item()
    # to avoid error in json.dumps in app.py
    classes = list()
    for unique_value in df[targetFeature].unique():
        classes.append(unique_value.item())
    classes.sort()

    color_scale = list()
    for j, cat in enumerate(classes):
        color = 'rgba' + total_color_scale[j][3:-1] + ', 0.8)'
        color_scale.append(color)

    train_length = len(df)

    # Calculate the variables to create the root node on the nodes part of the tree
    id = uuid.uuid4().hex[:8]
    isRoot = True
    fatherId = None
    data_list = list(data.values)
    entropy_node = entropy([x[1:] for x in data_list], classes)
    classPurity = calculatePurity(data_list, classes)
    isLeaf = False
    nodeClass= None
    for i, item in enumerate(classPurity):
        if item == 1:
            isLeaf = True
            nodeClass = classes[i]
    hasChildren = False
    childs = []
    leftChild = None
    rightChild = None
    pathFromRoot = []
    chosenAttr = None
    threshold = None
    samples = round((len(df)/train_length)*100, 1)

    tree_dict = {
        'data': {
            'df': data.to_json(orient='split'),
            'targetFeature': targetFeature,
            'selectedFeatures': selectedFeatures,
            'classes': classes,
            'train_length': train_length,
            'color_scale': color_scale
        },
        'nodes': [
            {
                'id': id,
                'isRoot': isRoot,
                'fatherId': fatherId,
                'classPurity': classPurity,
                'isLeaf': isLeaf,
                'class': nodeClass,
                'hasChildren': hasChildren,
                'childs': childs,
                'leftChild': leftChild,
                'rightChild': rightChild,
                'pathFromRoot': pathFromRoot,
                'chosenAttr': chosenAttr,
                'threshold': threshold,
                'entropy': entropy_node,
                'samples': samples
            }
        ],
        'clicks' : {
                'create' : create_btn,
                'delete' : delete_btn,
                'forceLeaf' : forceLeaf_btn,
                'resetLeaf' : resetLeaf_btn
        }
    }

    return tree_dict

def calculatePurity(data_list, classes):
    S = len(data_list)
    if S == 0:
        return 0
    num_classes = [0 for i in classes]
    for row in data_list:
        classIndex = list(classes).index(row[-1])
        num_classes[classIndex] += 1
    num_classes = [x / S for x in num_classes]
    return num_classes

def setIsLeafValue(tree, nodeId, isLeafValue, create_btn, delete_btn, forceLeaf_btn, resetLeaf_btn):
    """
    Function that sets the isLeaf value to the given value.
    It can only happen in the following situations:
    1.- isLeaf is False for the NodeId and isLeafValue given is True.
        This can only be possible if the NodeId has not children yet
    2.- isLeaf is True for the NodeId and isLeafValue given is False.
        This can only be possible if purity is NOT 1 for any class value

    :param tree: Tree with all the nodes and data
    :param nodeId: Id of the tree node whose isLeaf attribute needs to be set to True
    :param isLeafValue: True/False to be set
    :param create_btn:
        number of clicks of CREATE button at the moment the tree is created or modified
    :param delete_btn:
        number of clicks of DELETE button at the moment the tree is created or modified
    :param forceLeaf_btn:
        number of clicks of FORCE LEAF button at the moment the tree is created or modified
    :param resetLeaf_btn:
        number of clicks of RESET LEAF button at the moment the tree is created or modified
    :return:
        This function returns a dictionary that represents a Tree with the node whose Id was
        given as input, with the isLeaf variable set to True or False.
        It also returns a result boolean value:
          - True --> isLeaf has been changed
          - False --> isLeaf has not been changed. Same tree returned
        Finally it returns a string, in case result if False with the reason of failure
    """
    result = True
    result_info = None

    # First step is to generate the dataframe of the given node and get all the Node dict
    myNode = None
    for node in tree['nodes']:
        if node['id'] == nodeId:
            myNode = node

    # If nodeId is not present in the tree, return error
    if myNode == None:
        result = False
        result_info = "Node does not exist"
        # Update clicks values
        tree['clicks']['create'] = create_btn
        tree['clicks']['delete'] = delete_btn
        tree['clicks']['forceLeaf'] = forceLeaf_btn
        tree['clicks']['resetLeaf'] = resetLeaf_btn
        return tree, result, result_info

    # If given value isLeafValue is True but nodeId has children already
    # then error must be given
    if isLeafValue and myNode['hasChildren']:
        result = False
        result_info = "Node has children. Cannot be a Leaf"
        # Update clicks values
        tree['clicks']['create'] = create_btn
        tree['clicks']['delete'] = delete_btn
        tree['clicks']['forceLeaf'] = forceLeaf_btn
        tree['clicks']['resetLeaf'] = resetLeaf_btn
        return tree, result, result_info

    # If given value isLeafValue is True but nodeId is already a Leaf
    # then nothing to do
    if isLeafValue and myNode['isLeaf']:
        result = False
        result_info = "Node is already a Leaf"
        # Update clicks values
        tree['clicks']['create'] = create_btn
        tree['clicks']['delete'] = delete_btn
        tree['clicks']['forceLeaf'] = forceLeaf_btn
        tree['clicks']['resetLeaf'] = resetLeaf_btn
        return tree, result, result_info

    # If given value isLeafValue is False but nodeId is already a Leaf
    # with any class value purity equal to 1, then I cannot set to isLeaf = False
    if isLeafValue == False:
        for item in myNode['classPurity']:
            if item == 1:
                result = False
                result_info = "Node has purity equal to 1. Cannot be set to not isLeaf"
                # Update clicks values
                tree['clicks']['create'] = create_btn
                tree['clicks']['delete'] = delete_btn
                tree['clicks']['forceLeaf'] = forceLeaf_btn
                tree['clicks']['resetLeaf'] = resetLeaf_btn
                return tree, result, result_info

    # If given value isLeafValue is False and nodeId has isLeaf to False already
    # then nothing to do
    if isLeafValue == False and myNode['isLeaf'] == False:
        result = False
        result_info = "Node is not a Leaf already"
        # Update clicks values
        tree['clicks']['create'] = create_btn
        tree['clicks']['delete'] = delete_btn
        tree['clicks']['forceLeaf'] = forceLeaf_btn
        tree['clicks']['resetLeaf'] = resetLeaf_btn
        return tree, result, result_info

    # In any other case, perform the change
    # If changing from Leaf to Non-Leaf we need to set Class to None
    if isLeafValue == False:
        myNode['class'] = None
    # If changing from Non-Leaf to Leaf, we need to set Class to the class
    # with higher probability in 'classPurity'
    else:
        classPurity = myNode['classPurity']
        higher_index = None
        higher_value = 0
        for i, value in enumerate(classPurity):
            if value >= higher_value:
                higher_value = value
                higher_index = i
        myNode['class'] = tree['data']['classes'][higher_index]

    myNode['isLeaf'] = isLeafValue
    # Update clicks values
    tree['clicks']['create'] = create_btn
    tree['clicks']['delete'] = delete_btn
    tree['clicks']['forceLeaf'] = forceLeaf_btn
    tree['clicks']['resetLeaf'] = resetLeaf_btn

    return tree, result, result_info

def getNodeInfo(tree, nodeId):
    """
    :param tree: Tree with all the nodes and data
    :param nodeId: Id of the tree node whose data is retreived
    :return:
        This function returns a dictionary containing the Node
        whose ID is given as input parameter
        It also returns a result boolean value:
          - True --> Node has been found and returned
          - False --> Node has not been found
        Finally it returns a string, in case result if False with the reason of failure
    """
    # Initiate returned variables
    result = True
    result_info = None

    # First step is to generate the dataframe of the given node and get all the Node dict
    myNode = None
    for node in tree['nodes']:
        if node['id'] == nodeId:
            myNode = node

    # If nodeId is not present in the tree, return error
    if myNode == None:
        result = False
        result_info = "Node does not exist"
        return None, result, result_info
    else:
        return myNode, result, result_info


def splitNodeBy(tree, nodeId, chosenAttr, create_btn, delete_btn, forceLeaf_btn, resetLeaf_btn):
    """
    :param tree: Tree with all the nodes and data
    :param nodeId: Id of the tree node to be splitted
    :param chosenAttr: This is the attribute to be used to partition the Node
    :param create_btn:
        number of clicks of CREATE button at the moment the tree is created or modified
    :param delete_btn:
        number of clicks of DELETE button at the moment the tree is created or modified
    :param forceLeaf_btn:
        number of clicks of FORCE LEAF button at the moment the tree is created or modified
    :param resetLeaf_btn:
        number of clicks of RESET LEAF button at the moment the tree is created or modified
    :return:
        This function returns a new tree containing two new nodes that result
        from splitting the given node using the chosen attribute according to
        C4.5 algorithm. That is, finding the value of chosen attribute that
        maximizes the Information Gain after the split.
        It also returns a result boolean value:
          - True --> split has been done, returned tree is the resulting new tree
          - False --> split has not been done, returned tree is the same tree
        Finally it returns a string, in case result if False with the reason of failure
    """
    # Initiate returned variables
    result = True
    result_info = None

    # First step is to generate the dataframe of the given node and get all the Node dict
    myNode = None
    for node in tree['nodes']:
        if node['id'] == nodeId:
            myNode = node

    # If nodeId is not present in the tree, return error
    if myNode == None:
        result = False
        result_info = "Node does not exist"
        # Update clicks values
        tree['clicks']['create'] = create_btn
        tree['clicks']['delete'] = delete_btn
        tree['clicks']['forceLeaf'] = forceLeaf_btn
        tree['clicks']['resetLeaf'] = resetLeaf_btn
        return tree, result, result_info

    # If node is Leaf, no partition can be done
    if myNode['isLeaf']:
        result = False
        result_info = "Node is a Leaf"
        # Update clicks values
        tree['clicks']['create'] = create_btn
        tree['clicks']['delete'] = delete_btn
        tree['clicks']['forceLeaf'] = forceLeaf_btn
        tree['clicks']['resetLeaf'] = resetLeaf_btn
        return tree, result, result_info

    # If node already has children, no partition can be done
    if myNode['hasChildren']:
        result = False
        result_info = "Node already splited"
        # Update clicks values
        tree['clicks']['create'] = create_btn
        tree['clicks']['delete'] = delete_btn
        tree['clicks']['forceLeaf'] = forceLeaf_btn
        tree['clicks']['resetLeaf'] = resetLeaf_btn
        return tree, result, result_info

    nodeData, _ , _ = getData(tree, nodeId)

    # Then obtain the threshold value for chosen attribute that partitions the node
    # obtaining a maximum Information Gain according to C4.5 algorithm
    classes = tree['data']['classes']
    train_length = tree['data']['train_length']
    threshold, splitted = splitAttribute(nodeData, chosenAttr, classes)

    # Create two new nodes
    leftUid = uuid.uuid4().hex[:8]
    leftEntropy = entropy([x[1:] for x in splitted[0]], classes)
    leftPurity = calculatePurity(splitted[0], classes)
    leftIsLeaf = False
    leftNodeClass = None
    for i, item in enumerate(leftPurity):
        if item == 1:
            leftIsLeaf = True
            leftNodeClass = classes[i]
    leftSamples = round((len(splitted[0])/train_length)*100, 1)

    rightUid = uuid.uuid4().hex[:8]
    rightEntropy = entropy([x[1:] for x in splitted[1]], classes)
    rightPurity = calculatePurity(splitted[1], classes)
    rightIsLeaf = False
    rightNodeClass = None
    for i, item in enumerate(rightPurity):
        if item == 1:
            rightIsLeaf = True
            rightNodeClass = classes[i]
    rightSamples = round((len(splitted[1])/train_length)*100, 1)

    nodeLeft = {
        'id': leftUid,
        'isRoot': False,
        'fatherId': nodeId,
        'classPurity': leftPurity,
        'isLeaf': leftIsLeaf,
        'class': leftNodeClass,
        'hasChildren': False,
        'childs': [],
        'leftChild': None,
        'rightChild': None,
        'pathFromRoot': myNode['pathFromRoot'] + [nodeId],
        'chosenAttr': None,
        'threshold': None,
        'entropy': leftEntropy,
        'samples': leftSamples
    }
    nodeRight = {
            'id': rightUid,
            'isRoot': False,
            'fatherId': nodeId,
            'classPurity': rightPurity,
            'isLeaf': rightIsLeaf,
            'class': rightNodeClass,
            'hasChildren': False,
            'childs': [],
            'leftChild': None,
            'rightChild': None,
            'pathFromRoot': myNode['pathFromRoot'] + [nodeId],
            'chosenAttr': None,
            'threshold': None,
            'entropy': rightEntropy,
            'samples': rightSamples
        }

    # Add the two new nodes to the childs of all its ancestrors
    # Update own node with hasChildren, left and right childs, chosen attribute and threshold value
    if myNode['isRoot']:
        for node in tree['nodes']:
            if node['id'] == nodeId:
                node['childs'] = node['childs'] + [leftUid, rightUid]
                node['hasChildren'] = True
                node['leftChild'] = leftUid
                node['rightChild'] = rightUid
                node['chosenAttr'] = chosenAttr
                node['threshold'] = threshold
    else:
        for node in tree['nodes']:
            if node['id'] in myNode['pathFromRoot'] or node['id'] == nodeId:
                node['childs'] = node['childs'] + [leftUid, rightUid]
                if node['id'] == nodeId:
                    node['hasChildren'] = True
                    node['leftChild'] = leftUid
                    node['rightChild'] = rightUid
                    node['chosenAttr'] = chosenAttr
                    node['threshold'] = threshold

    # Add the two new ones to the tree
    tree['nodes'] = tree['nodes'] + [nodeLeft, nodeRight]
    # Update clicks values
    tree['clicks']['create'] = create_btn
    tree['clicks']['delete'] = delete_btn
    tree['clicks']['forceLeaf'] = forceLeaf_btn
    tree['clicks']['resetLeaf'] = resetLeaf_btn

    return tree, result, result_info

def getData(tree, nodeId):
    """
    This function takes a tree dictionary as input, a nodeId and returns the dataframe
    that corresponds to the node with id = nodeId by building it from root dataframe
    and applying the filters of all nodes from root to the node with id = nodeId
    :param tree: Dictionary with the tree information
    :param nodeId: Id of the node whose dataframe needs to be build
    :return:
        dataframe of the node with id = nodeId
        It also returns a result boolean value:
          - True --> Node has been found and tree is returned
          - False --> Node has not been found
        Finally it returns a string, in case result if False with the reason of failure
    """
    # Initiate returned variables
    result = True
    result_info = None

    # First step is to generate the dataframe of the given node and get all the Node dict
    myNode = None
    for node in tree['nodes']:
        if node['id'] == nodeId:
            myNode = node

    # If nodeId is not present in the tree, return error
    if myNode == None:
        result = False
        result_info = "Node does not exist"
        return None, result, result_info

    df = pd.read_json(tree['data']['df'], orient='split')

    if myNode['isRoot']:
        return df, result, result_info
    else:
        pathFromRoot = myNode['pathFromRoot']
        filterList = []
        for id in pathFromRoot:
            for node in tree['nodes']:
                if node['id'] == id:
                    filterList.append((node['chosenAttr'],node['threshold'],node['leftChild'],node['rightChild']))

        for filterItem in filterList:
            itemChosenAttr = filterItem[0]
            itemThreshold = filterItem[1]
            itemLeftChild = filterItem[2]
            itemRightChild = filterItem[3]
            if itemLeftChild in pathFromRoot or itemLeftChild == myNode['id']:
                df = df[df[itemChosenAttr] <= itemThreshold]
                pass
            else:
                df = df[df[itemChosenAttr] > itemThreshold]
                pass

        return df, result, result_info

def deleteNodes(tree, nodeId, create_btn, delete_btn, forceLeaf_btn, resetLeaf_btn):
    """
    This function deletes all the nodes below the given Node.
    It also updates all its ancestrors to remove the deleted nodes
    from their childs list.
    A node without children will not produce any change.

    :param tree: Dictionary with the tree information
    :param nodeId: Id of the node whose childs need to be deleted
    :param create_btn:
        number of clicks of CREATE button at the moment the tree is created or modified
    :param delete_btn:
        number of clicks of DELETE button at the moment the tree is created or modified
    :param forceLeaf_btn:
        number of clicks of FORCE LEAF button at the moment the tree is created or modified
    :param resetLeaf_btn:
        number of clicks of RESET LEAF button at the moment the tree is created or modified
    :return:
        This function returns a new tree with all the nodes except the
        ones below the node whose id is given as input
        It also returns a result boolean value:
          - True --> deletion has been done, returned tree is the resulting new tree
          - False --> deletion has not been done, returned tree is the same tree
        Finally it returns a string, in case result if False with the reason of failure
    """
    # Initiate returned variables
    result = True
    result_info = None

    # Get the node whose children need to be deleted
    for node in tree['nodes']:
        if node['id'] == nodeId:
            myNode = node

    # If node has no children, no deletion can be done
    if not myNode['hasChildren']:
        result = False
        result_info = "Node is a Leaf"
        # Update clicks values
        tree['clicks']['create'] = create_btn
        tree['clicks']['delete'] = delete_btn
        tree['clicks']['forceLeaf'] = forceLeaf_btn
        tree['clicks']['resetLeaf'] = resetLeaf_btn
        return tree, result, result_info

    # Get childs list
    myChilds = myNode['childs']

    # Iterate over the list of nodes from the tree
    # If the id of the node is on myChilds list
    # then remove the node from the tree
    nodeList = tree['nodes'].copy()
    for node in nodeList:
        if node['id'] in myChilds:
            tree['nodes'].remove(node)

    # Now we need to update all the myNode ancestrors to
    # remove the ids of the deleted nodes from their childs list
    # We need also to set myNode childs list to empty list
    # as well as set to None left and right childs. Threshold value and
    # chosen attriute must also be set to None
    for node in tree['nodes']:
        if node['id'] == nodeId:
            node['hasChildren'] = False
            node['childs'] = []
            node['leftChild'] = None
            node['rightChild'] = None
            node['chosenAttr'] = None
            node['threshold'] = None
        else:
            if node['id'] in myNode['pathFromRoot']:
                for childId in myChilds:
                    node['childs'].remove(childId)

    # Update clicks values
    tree['clicks']['create'] = create_btn
    tree['clicks']['delete'] = delete_btn
    tree['clicks']['forceLeaf'] = forceLeaf_btn
    tree['clicks']['resetLeaf'] = resetLeaf_btn
    return tree, result, result_info

def isTreeClassifier(tree):
    """
    This function checks if the tree can be used for classification.
    This will only be possible if all terminating nodes (that is, nodes
    without children) are marked as Leaf.

    :param tree: Dictionary with the tree information
    :return: Boolean
                - True: if all terminating nodes (not having children) are leaf
                - False: if any terminating node (not having children) is not leaf
    """
    # Initiate returned variables
    result = True

    # Get all the nodes in tree
    for node in tree['nodes']:
        if not node['hasChildren']:
            if not node['isLeaf']:
                result = False
                break

    return result

def executeClassification(tree_jsonified, df_jsonified, index):
    """
    This function takes a tree as input and returns the classification results
    according to that tree.

    :param tree: Dictionary with the tree information
    :param df: Original DF
    :param index: List with index for test observations
    :return: Tuple containing:
                - y_true: list with the real classes for the test observations
                - y_pred: list with the predicted classes for the test observations
                - classes: list with the different classes
                - labels: list with the labels for the classes
                - own_tree_dot: DOT representation of tree
    """
    # Get the dataframe wiht only the test observations and the selected features
    index.sort()
    df_full = pd.read_json(df_jsonified, orient='split')
    df_test = df_full[df_full.index.isin(index)]
    tree_dict = json.loads(tree_jsonified)
    targetFeature = tree_dict['data']['targetFeature']
    y_true = df_test[targetFeature].tolist()
    selectedFeatures = tree_dict['data']['selectedFeatures']
    selectedFeatures.remove('index')
    df_test_features = df_test[selectedFeatures]

    # Now build the tree information in dictionary form to be passed to classify function
    tree_info = dict()
    for node in tree_dict['nodes']:
        tree_info[node['id']] = {
                                'isLeaf': node['isLeaf'],
                                'class': node['class'],
                                'chosenAttr': node['chosenAttr'],
                                'threshold': node['threshold'],
                                'rightChild': node['rightChild'],
                                'leftChild': node['leftChild']
                              }

    # Finally, get Root Node id to pass as first node to classify function
    rootNode = next(node for node in tree_dict['nodes'] if node['isRoot'])
    rootNodeid = rootNode['id']

    # Build a list with all the rows of the df_test_features as Pandas Series objects
    row_list = [df_test_features.loc[i,:] for i in df_test_features.index.tolist()]
    y_pred = [classifyObservation(row, tree_info, rootNodeid) for row in row_list]

    # Build classes and labels
    classes = tree_dict['data']['classes']
    labels = ['class_{}'.format(item) for item in classes]

    # Generate DOT representation
    own_tree_dot = export_to_dot(tree_dict)

    return y_true, y_pred, classes, labels, own_tree_dot

def classifyObservation(df_row, tree_info, nodeId):
    """
    This is a recursive function that classifies an observation following a
    decision tree from the specified node.

    :param df_row: This is a row from the df with all the test observations. It is a Pandas Series object
    :param tree_info: Information about the tree in a dictionary form wih dictionary also as value.
                      One key for each nodeId with the rest of needed attributes also as keys of the inner dictionary
    :param node: This is the Id of the node in which the observation is currently being analyzed
    :return: The return value is a value corresponing with the class of the given node
    """
    if tree_info[nodeId]['isLeaf']:
        return tree_info[nodeId]['class']
    else:
        feature = tree_info[nodeId]['chosenAttr']
        threshold = tree_info[nodeId]['threshold']
        if df_row[feature] > threshold:
            return classifyObservation(df_row, tree_info, tree_info[nodeId]['rightChild'])
        else:
            return classifyObservation(df_row, tree_info, tree_info[nodeId]['leftChild'])


def export_to_dot(tree_dict):
    """
    This function receives a jsonified tree_dict (containing data and tree information)
    and outputs an string that represents that tree in a dot format to be rendered using
    Graphviz tool.

    :param tree_dict_jsonified: Jsonified version of the complete tree_dict stored in a hidden DIV
    :return: string containing DOT version of the tree to be rendered using Graphviz
    """
    tree = tree_dict['nodes']
    color_list = tree_dict['data']['color_scale']
    hex_color_list = list()
    for color in color_list:
        rgb_tuple = rgba_to_rgb(color)
        hex = wc.rgb_to_hex(rgb_tuple)
        hex_color_list.append(hex)

    classes = tree_dict['data']['classes']

    # Generate dot file general parts
    dot_tree = 'digraph Tree {\n'
    dot_tree += 'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;\n'
    dot_tree += 'edge [fontname=helvetica] ;\n'

    # Iterate over nodes and generate nodes and edges items
    for j, node in enumerate(tree):
        purity = '['
        for item in node['classPurity']:
            purity += str(round(item, 2)) + ','
        purity = purity.rstrip(',')
        purity += ']'
        if node['hasChildren']:
            dot_tree += '{0} [label="{1} <= {2}\\nentropy = {3}\\nsamples = {4}%\\nvalue = '.format(str(int(node['id'], 16)), node['chosenAttr'],
                                                                    str(round(node['threshold'], 2)), str(round(node['entropy'], 3)),
                                                                    str(node['samples'])) + \
                        purity + '", fillcolor="#ffffff"] ;\n'
        else:
            if node['class'] is not None:
                my_hex = hex_color_list[classes.index(int(node['class']))]
                dot_tree += '{0} [label="entropy = {1}\\nsamples = {2}%\\nvalue = '.format(str(int(node['id'], 16)), str(round(node['entropy'], 3)),
                                                                                           str(node['samples'])) + purity + \
                            '\\nclass = {0}", fillcolor="{1}"] ;\n'.format(str(node['class']), my_hex)
            else:
                dot_tree += '{0} [label="entropy = {1}\\nsamples = {2}%\\nvalue = '.format(str(int(node['id'], 16)), str(round(node['entropy'], 3)),
                                                                                           str(node['samples'])) + purity + \
                            ' ", fillcolor="#ffffff"] ;\n'

        if node['fatherId'] is not None:
            if len(node['pathFromRoot']) == 1:
                if tree[0]['leftChild'] == node['id']:
                    dot_tree += '{0} -> {1} [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n'. \
                        format(str(int(node['fatherId'], 16)), str(int(node['id'], 16)))
                else:
                    dot_tree += '{0} -> {1} [labeldistance = 2.5, labelangle = -45, headlabel = "False"] ;\n'. \
                        format(str(int(node['fatherId'], 16)), str(int(node['id'], 16)))
            else:
                dot_tree += '{0} -> {1} ;\n'.format(str(int(node['fatherId'], 16)), str(int(node['id'], 16)))

    dot_tree += '}'

    return dot_tree

def rgba_to_rgb(rgba_str):
    """
    This function takes an string of format rgba(RED, GREEN, BLUE, ALPHA) and returns
    the equivalent RGB triplet considering a white background

    :param rgba_str: rgba(RED, GREEN, BLUE, ALPHA)
    :return: rgb(RED, GREEN, BLUE)
    """
    rgba_tuple = tuple(float(i) for i in rgba_str.strip('rgba(').rstrip(')').split(','))
    red = int(rgba_tuple[0])
    green = int(rgba_tuple[1])
    blue = int(rgba_tuple[2])
    alpha = rgba_tuple[3]

    # We assume white background (255, 255, 255)
    new_red = int(red * alpha + 255 * (1 - alpha))
    new_green = int(green * alpha + 255 * (1 - alpha))
    new_blue = int(blue * alpha + 255 * (1 - alpha))

    return (new_red, new_green, new_blue)
