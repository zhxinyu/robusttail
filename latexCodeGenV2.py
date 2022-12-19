import numpy as np 
import pandas as pd

def getSignificanceNExponent (value: float):
        exponent = np.floor(np.log(value)/np.log(10))
        return (value/10**exponent, exponent)
    
def tableFiveOneUnit(targetColumns, dataSource, EstimatedUpperBounds, RelativeRatios, CoverageProbabilities):
    content = r""
    for i, targetColumn in enumerate(targetColumns):
        significance0, exponent0 = getSignificanceNExponent(EstimatedUpperBounds[0][i])
        if i == 0:
            content+="\multirow{{{:}}}{{*}}{{{:}}}".format(len(targetColumns), dataSource.capitalize())
        if targetColumn == '(0,CHI2)':
            content+=r"&$(0,\chi^2)$" 
        elif targetColumn == '(1,CHI2)':
            content+=r"&$(1,\chi^2)$"
        elif targetColumn == '(2,CHI2)':
            content+=r"&$(2,\chi^2)$"
        elif targetColumn == '(0,KS)':
            content+=r"&$(0,\text{{KS}})$"
        elif targetColumn == '(1,KS)':
            content+=r"&$(1,\text{{KS}})$"
        elif targetColumn == '(2,KS)':
            content+=r"&$(2,\text{{KS}})$"
        else:
            print(targetColumn)
            assert False
        content+=r"& ${:.2f}$ & ${:.2f}\times 10^{{{:d}}}$ & ${:2g}$".format(
            RelativeRatios[0][i], significance0, int(exponent0),CoverageProbabilities[0][i])
        if i==len(targetColumns)-1: 
            content+="\\\\"
        else:
            content+="\\\\\n"        
    return content


def tableFiveOne(groupby_object1:pd.core.groupby.generic.DataFrameGroupBy, 
                 dataSources: list, nDatas: list, percentageLHS: list, thresholdPercentage:float,
                 targetColumns: list):
    trueValues = []
    contents = []
    for dataSource in dataSources:
        RelativeRatios = [ 0 for _ in range(len(nDatas))]
        CoverageProbabilities = [ 0 for _ in range(len(nDatas))]
        EstimatedUpperBounds = [ 0 for _ in range(len(nDatas))]
        for i, nData in enumerate(nDatas):
            currKeyChoice = (dataSource, nData, percentageLHS, thresholdPercentage)
            trueValues.append(groupby_object1.get_group(currKeyChoice)['trueValue'].unique()[0])
            EstimatedUpperBounds[i]= groupby_object1.get_group(currKeyChoice)[targetColumns].mean().tolist()
            RelativeRatios[i] = (groupby_object1.get_group(currKeyChoice)[targetColumns].mean().values/trueValues[-1]).tolist()
            CoverageProbabilities[i]=(groupby_object1.get_group(currKeyChoice)[targetColumns].values>trueValues[-1]).mean(axis=0).tolist()
        contents.append(tableFiveOneUnit(targetColumns, dataSource, EstimatedUpperBounds, RelativeRatios, CoverageProbabilities))
    ## trueValue should be same among different percentageLHS.         
    assert np.unique(trueValues).size == 1         
    return ("\\hline \n".join(contents), trueValues[0])

    
def getTableOne(go1:pd.core.groupby.generic.DataFrameGroupBy, 
                 dataSources: list, nDatas: list, percentageLHS: float, thresholdPercentage:float,
                 targetColumns: list, 
                 title:str, label: str, scalebox: float):
    content, trueValue = tableFiveOne(go1, dataSources, nDatas, percentageLHS, thresholdPercentage, targetColumns)
    latexTable= r'''
\begin{table}[ht]
    \centering'''+\
    r'''\scalebox{{{:}}}{{'''.format(scalebox)+\
    r'''\begin{tabular}{cc|ccc}
    \toprule
    \hline
    '''+\
    r'''Data Source & Constraint Setting & Relative Error & Upper Bound & Coverage Probability\\\hline'''+"\n"+\
    content+\
    r'''
    \hline
    \bottomrule
    \end{tabular}}'''+\
    '''\caption{{{:}}}
    \label{{{:}}}
\end{{table}}
    '''.format(title+"The true value is {:}.".format(trueValue), label)
    return latexTable


def tableFiveTwoUnit(trueValue, targetColumns, dataSource, EstimatedUpperBounds, RelativeRatios, CoverageProbabilities):
    content = r""
    for i, targetColumn in enumerate(targetColumns):
        significance0, exponent0 = getSignificanceNExponent(EstimatedUpperBounds[0][i])
        if i == 0:
            content+="\multirow{{{:}}}{{*}}{{{:}}}".format(len(targetColumns), dataSource.capitalize()+" w. true quantile point {:.2f}.".format(trueValue))
        if targetColumn == '(0,CHI2)':
            content+=r"&$(0,\chi^2)$" 
        elif targetColumn == '(1,CHI2)':
            content+=r"&$(1,\chi^2)$"
        elif targetColumn == '(2,CHI2)':
            content+=r"&$(2,\chi^2)$"
        elif targetColumn == '(0,KS)':
            content+=r"&$(0,\text{{KS}})$"
        elif targetColumn == '(1,KS)':
            content+=r"&$(1,\text{{KS}})$"
        elif targetColumn == '(2,KS)':
            content+=r"&$(2,\text{{KS}})$"
        else:
            print(targetColumn)
            assert False
        content+=r"& ${:.2f}$ & ${:.2f}\times 10^{{{:d}}}$ & ${:2g}$".format(
            RelativeRatios[0][i], significance0, int(exponent0),CoverageProbabilities[0][i])
        if i==len(targetColumns)-1: 
            content+="\\\\"
        else:
            content+="\\\\\n"        
    return content


def tableFiveTwo(groupby_object1:pd.core.groupby.generic.DataFrameGroupBy, 
                 dataSources: list, nDatas: list, percentageLHS: list, thresholdPercentage:float,
                 targetColumns: list):
    trueValues = []
    contents = []
    for dataSource in dataSources:
        RelativeRatios = [ 0 for _ in range(len(nDatas))]
        CoverageProbabilities = [ 0 for _ in range(len(nDatas))]
        EstimatedUpperBounds = [ 0 for _ in range(len(nDatas))]
        for i, nData in enumerate(nDatas):
            currKeyChoice = (dataSource, nData, percentageLHS, thresholdPercentage)
            trueValues.append(groupby_object1.get_group(currKeyChoice)['trueValue'].unique()[0])
            EstimatedUpperBounds[i]= groupby_object1.get_group(currKeyChoice)[targetColumns].mean().tolist()
            RelativeRatios[i] = (groupby_object1.get_group(currKeyChoice)[targetColumns].mean().values/trueValues[-1]).tolist()
            CoverageProbabilities[i]=(groupby_object1.get_group(currKeyChoice)[targetColumns].values>trueValues[-1]).mean(axis=0).tolist()
        contents.append(tableFiveTwoUnit(trueValues[-1], targetColumns, dataSource, EstimatedUpperBounds, RelativeRatios, CoverageProbabilities))
    return ("\\hline \n".join(contents), trueValues[0])

    
def getTableTwo(go3:pd.core.groupby.generic.DataFrameGroupBy, 
                 dataSources: list, nDatas: list, percentageLHS: float, thresholdPercentage:float,
                 targetColumns: list, 
                 title:str, label: str, scalebox: float):
    content, trueValue = tableFiveTwo(go3, dataSources, nDatas, percentageLHS, thresholdPercentage, targetColumns)
    latexTable= r'''
\begin{table}[ht]
    \centering'''+\
    r'''\scalebox{{{:}}}{{'''.format(scalebox)+\
    r'''\begin{tabular}{cc|ccc}
    \toprule
    \hline
    '''+\
    r'''Data Source & Constraint Setting & Relative Error & Upper Bound & Coverage Probability \\\hline'''+"\n"+\
    content+\
    r'''
    \hline
    \bottomrule
    \end{tabular}}'''+\
    '''\caption{{{:}}}
    \label{{{:}}}
\end{{table}}
    '''.format(title, label)
    return latexTable
    
def tableFiveThreeUnit(thresholds, dataSource,
                      EstimatedUpperBounds, RelativeRatios, CoverageProbabilities):
    
    content = r""
    for i, threshold in enumerate(thresholds):
        significance0, exponent0 = getSignificanceNExponent(EstimatedUpperBounds[i][0][0])  
        if i == 0:
            content+="\multirow{{{:}}}{{*}}{{{:}}}".format(len(thresholds)+1, dataSource.capitalize())

        content+=r"&${:.3f}$ & ${:.3f}$ & ${:.2f}\times 10^{{{:d}}}$ & ${:2g}$".format(
            threshold, 
            RelativeRatios[i][0][0], significance0, int(exponent0), CoverageProbabilities[i][0][0])
        content+="\\\\\n"

    significance0, exponent0 = getSignificanceNExponent(EstimatedUpperBounds[-1][0][0])  

    content+=r"&${:}$ & ${:.3f}$ & ${:.2f}\times 10^{{{:d}}}$ & ${:2g}$".format(
        thresholds, 
        RelativeRatios[-1][0][0], significance0, int(exponent0), CoverageProbabilities[-1][0][0])
    content+="\\\\"
    return content


def tableFiveThreeUnitQE(thresholds, dataSource,
                      EstimatedUpperBounds, RelativeRatios, CoverageProbabilities, trueValue):
    
    content = r""
    for i, threshold in enumerate(thresholds):
        significance0, exponent0 = getSignificanceNExponent(EstimatedUpperBounds[i][0][0])  
        if i == 0:
            content+="\multirow{{{:}}}{{*}}{{{:}}}".format(len(thresholds)+1, dataSource.capitalize()+" w. true quantile point {:.2f}".format(trueValue))

        content+=r"&${:.3f}$ & ${:.3f}$ & ${:.2f}\times 10^{{{:d}}}$ & ${:2g}$".format(
            threshold, 
            RelativeRatios[i][0][0], significance0, int(exponent0), CoverageProbabilities[i][0][0])
        content+="\\\\\n"

    significance0, exponent0 = getSignificanceNExponent(EstimatedUpperBounds[-1][0][0])  

    content+=r"&${:}$ & ${:.3f}$ & ${:.2f}\times 10^{{{:d}}}$ & ${:2g}$".format(
        thresholds, 
        RelativeRatios[-1][0][0], significance0, int(exponent0), CoverageProbabilities[-1][0][0])
    content+="\\\\"
    return content



# keyChoice1 = ('gamma', 500, -1, thresholdPercentage)


def tableFiveThree(groupby_object1:pd.core.groupby.generic.DataFrameGroupBy, 
                   groupby_object2:pd.core.groupby.generic.DataFrameGroupBy, 
                   dataSources: list, nDatas: list, percentageLHS: float, thresholdPercentage: float,
                   targetColumns: list):
    trueValues = []
    contents = []
    for dataSource in dataSources:
        numMultiThreshold = 4
        RelativeRatios = [[ 0 for _ in range(len(nDatas))] for _ in range(numMultiThreshold+1)]
        CoverageProbabilities = [[ 0 for _ in range(len(nDatas))] for _ in range(numMultiThreshold+1)]
        EstimatedUpperBounds = [[ 0 for _ in range(len(nDatas))] for _ in range(numMultiThreshold+1)]
        for i in range(numMultiThreshold):
            for j, nData in enumerate(nDatas):
                currKeyChoice = (dataSource, nData, percentageLHS, round(thresholdPercentage+0.1*i,1))
                trueValues.append(groupby_object1.get_group(currKeyChoice)['trueValue'].unique()[0])
                EstimatedUpperBounds[i][j]= groupby_object1.get_group(currKeyChoice)[targetColumns].mean().tolist()
                RelativeRatios[i][j] = (groupby_object1.get_group(currKeyChoice)[targetColumns].mean().values/trueValues[-1]).tolist()
                CoverageProbabilities[i][j]=(groupby_object1.get_group(currKeyChoice)[targetColumns].values>trueValues[-1]).mean(axis=0).tolist()
        for j, nData in enumerate(nDatas):
            currKeyChoice = (dataSource, nData, percentageLHS, thresholdPercentage)
            trueValues.append(groupby_object2.get_group(currKeyChoice)['trueValue'].unique()[0])
            EstimatedUpperBounds[numMultiThreshold][j]= groupby_object2.get_group(currKeyChoice)[targetColumns].mean().tolist()
            RelativeRatios[numMultiThreshold][j] = (groupby_object2.get_group(currKeyChoice)[targetColumns].mean().values/trueValues[-1]).tolist()
            CoverageProbabilities[numMultiThreshold][j]=(groupby_object2.get_group(currKeyChoice)[targetColumns].values>trueValues[-1]).mean(axis=0).tolist()        
            
        thresholds = [round(thresholdPercentage+0.1*i,1) for i in range(numMultiThreshold)]            
        contents.append(tableFiveThreeUnit(thresholds, dataSource, EstimatedUpperBounds, RelativeRatios, CoverageProbabilities))
        
    ## trueValue should be same among different thresholdPercentage.         
    assert np.unique(trueValues).size == 1         
    return ("\\hline \n".join(contents), trueValues[0])

def tableFiveThreeQE(groupby_object1:pd.core.groupby.generic.DataFrameGroupBy, 
                   groupby_object2:pd.core.groupby.generic.DataFrameGroupBy, 
                   dataSources: list, nDatas: list, percentageLHS: float, thresholdPercentage: float,
                   targetColumns: list):
    contents = []
    for dataSource in dataSources:
        trueValues = []
        numMultiThreshold = 4
        RelativeRatios = [[ 0 for _ in range(len(nDatas))] for _ in range(numMultiThreshold+1)]
        CoverageProbabilities = [[ 0 for _ in range(len(nDatas))] for _ in range(numMultiThreshold+1)]
        EstimatedUpperBounds = [[ 0 for _ in range(len(nDatas))] for _ in range(numMultiThreshold+1)]
        for i in range(numMultiThreshold):
            for j, nData in enumerate(nDatas):
                currKeyChoice = (dataSource, nData, percentageLHS, round(thresholdPercentage+0.1*i,1))
                trueValues.append(groupby_object1.get_group(currKeyChoice)['trueValue'].unique()[0])
                EstimatedUpperBounds[i][j]= groupby_object1.get_group(currKeyChoice)[targetColumns].mean().tolist()
                RelativeRatios[i][j] = (groupby_object1.get_group(currKeyChoice)[targetColumns].mean().values/trueValues[-1]).tolist()
                CoverageProbabilities[i][j]=(groupby_object1.get_group(currKeyChoice)[targetColumns].values>trueValues[-1]).mean(axis=0).tolist()
        for j, nData in enumerate(nDatas):
            currKeyChoice = (dataSource, nData, percentageLHS, thresholdPercentage)
            trueValues.append(groupby_object2.get_group(currKeyChoice)['trueValue'].unique()[0])
            EstimatedUpperBounds[numMultiThreshold][j]= groupby_object2.get_group(currKeyChoice)[targetColumns].mean().tolist()
            RelativeRatios[numMultiThreshold][j] = (groupby_object2.get_group(currKeyChoice)[targetColumns].mean().values/trueValues[-1]).tolist()
            CoverageProbabilities[numMultiThreshold][j]=(groupby_object2.get_group(currKeyChoice)[targetColumns].values>trueValues[-1]).mean(axis=0).tolist()        
        ## trueValue should be same among different thresholdPercentage.         
        assert np.unique(trueValues).size == 1                 
        thresholds = [round(thresholdPercentage+0.1*i,1) for i in range(numMultiThreshold)]            
        contents.append(tableFiveThreeUnitQE(thresholds, dataSource, EstimatedUpperBounds, RelativeRatios, CoverageProbabilities, trueValues[0]))
    return "\\hline \n".join(contents)

def getTableThree(go1:pd.core.groupby.generic.DataFrameGroupBy, 
                  go2:pd.core.groupby.generic.DataFrameGroupBy, 
                  dataSources: list, nDatas: list, percentageLHS: float, thresholdPercentage:float,
                  targetColumns: list, 
                  title:str, label: str, scalebox: float, isQE = False):
    
    if isQE:
        content= tableFiveThreeQE(go1, go2, dataSources, nDatas, percentageLHS, thresholdPercentage, targetColumns)            
    else:
        content, trueValue = tableFiveThree(go1, go2, dataSources, nDatas, percentageLHS, thresholdPercentage, targetColumns)    

    latexTable= r'''
\begin{table}[ht]
    \centering'''+\
    r'''\scalebox{{{:}}}{{'''.format(scalebox)+\
    r'''\begin{tabular}{cc|ccc}
    \toprule
    \hline
    '''+\
    r'''Data Source & Constraint Setting & Relative Error & Upper Bound & Coverage Probability \\\hline'''+"\n"+\
    content+\
    r'''
    \hline
    \bottomrule
    \end{tabular}}'''+\
    '''\caption{{{:}}}
    \label{{{:}}}
\end{{table}}
    '''.format(title + ("The true value is {:}.".format(0.5) if not isQE else ""), label)
    
    return latexTable
        
def tableFiveFourUnit(percentageLHSs, dataSource,
                      EstimatedUpperBounds, RelativeRatios, CoverageProbabilities):
    content = r""
    for i, percentageLHS in enumerate(percentageLHSs):
        significance0, exponent0 = getSignificanceNExponent(EstimatedUpperBounds[i][0][0])
        significance1, exponent1 = getSignificanceNExponent(EstimatedUpperBounds[i][0][1])    
        if i == 0:
            content+="\multirow{{{:}}}{{*}}{{{:}}}".format(len(percentageLHSs), dataSource.capitalize())
            
        content+=r"&${:.3f}$ & ${:.3f}$ & ${:.2f}\times 10^{{{:d}}}$ & ${:2g}$ & ${:.3f}$ & ${:.2f}\times 10^{{{:d}}}$ & ${:2g}$".format(
            percentageLHS, 
            RelativeRatios[i][0][0], significance0, int(exponent0), CoverageProbabilities[i][0][0],
            RelativeRatios[i][0][1], significance1, int(exponent1), CoverageProbabilities[i][0][1])
        if i==len(percentageLHSs)-1: 
            content+="\\\\"
        else:
            content+="\\\\\n"
    return content


# keyChoice1 = ('gamma', 500, -1, thresholdPercentage)


def tableFiveFour(groupby_object1:pd.core.groupby.generic.DataFrameGroupBy, 
                  dataSources: list, nDatas: list, percentageLHSs: list, thresholdPercentage:float,
                  targetColumns: list):
    trueValues = []
    contents = []
    for dataSource in dataSources:
        RelativeRatios = [[ 0 for _ in range(len(nDatas))] for _ in range(len(percentageLHSs))]
        CoverageProbabilities = [[ 0 for _ in range(len(nDatas))] for _ in range(len(percentageLHSs))]
        EstimatedUpperBounds = [[ 0 for _ in range(len(nDatas))] for _ in range(len(percentageLHSs))]
        for i, percentageLHS in enumerate(percentageLHSs):
            for j, nData in enumerate(nDatas):
                currKeyChoice = (dataSource, nData, percentageLHS, thresholdPercentage)
                trueValues.append(groupby_object1.get_group(currKeyChoice)['trueValue'].unique()[0])
                EstimatedUpperBounds[i][j]= groupby_object1.get_group(currKeyChoice)[targetColumns].mean().tolist()
                RelativeRatios[i][j] = (groupby_object1.get_group(currKeyChoice)[targetColumns].mean().values/trueValues[-1]).tolist()
                CoverageProbabilities[i][j]=(groupby_object1.get_group(currKeyChoice)[targetColumns].values>trueValues[-1]).mean(axis=0).tolist()
        contents.append(tableFiveFourUnit(percentageLHSs, dataSource, EstimatedUpperBounds, RelativeRatios, CoverageProbabilities))
    ## trueValue should be same among different percentageLHS.         
    assert np.unique(trueValues).size == 1         
    return ("\\hline \n".join(contents), trueValues[0])


def getTableFour(go1:pd.core.groupby.generic.DataFrameGroupBy, 
                 dataSources: list, nDatas: list, percentageLHSs: list, thresholdPercentage:float,
                 targetColumns: list, 
                 title:str, label: str, scalebox: float):
    content, trueValue = tableFiveFour(go1, dataSources, nDatas, percentageLHSs, thresholdPercentage, targetColumns)
    latexTable= r'''
\begin{table}[ht]
    \centering'''+\
    r'''\scalebox{{{:}}}{{'''.format(scalebox)+\
    r'''\begin{tabular}{cc|ccc|ccc}
    \toprule
    \hline
    \multicolumn{1}{c}{} & \multicolumn{1}{c}{} &'''+\
    r'''\multicolumn{3}{|c|}{{$(2,\chi^2)$}} & \multicolumn{3}{|c}{{$(2, \textup{KS})$}}''' +r'''\\
    Data Source & LHS Quantitle & Relative Error & Upper Bound & Coverage Probability & Relative Error & Upper Bound & Coverage Probability \\\hline'''+"\n"+\
    content+\
    r'''
    \hline
    \bottomrule
    \end{tabular}}'''+\
    '''\caption{{{:}}}
    \label{{{:}}}
\end{{table}}
    '''.format(title+"The true value is {:}.".format(trueValue), label)
    return latexTable



def tableFiveUnit(percentageLHSs, dataSource,
                      EstimatedUpperBounds, RelativeRatios, CoverageProbabilities):
    content = r""
    for i, percentageLHS in enumerate(percentageLHSs):
        significance0, exponent0 = getSignificanceNExponent(EstimatedUpperBounds[i][0][0]) 
        if i == 0:
            content+="\multirow{{{:}}}{{*}}{{{:}}}".format(len(percentageLHSs), dataSource.capitalize())
            
        content+=r"&${:.3f}$ & ${:.3f}$ & ${:.2f}\times 10^{{{:d}}}$ & ${:2g}$".format(
            percentageLHS, 
            RelativeRatios[i][0][0], significance0, int(exponent0), CoverageProbabilities[i][0][0])
        if i==len(percentageLHSs)-1: 
            content+="\\\\"
        else:
            content+="\\\\\n"
    return content


def tableFive(groupby_object1:pd.core.groupby.generic.DataFrameGroupBy, 
                  dataSources: list, nDatas: list, percentageLHSs: list,
                  targetColumns: list):
    trueValues = []
    contents = []
    for dataSource in dataSources:
        RelativeRatios = [[ 0 for _ in range(len(nDatas))] for _ in range(len(percentageLHSs))]
        CoverageProbabilities = [[ 0 for _ in range(len(nDatas))] for _ in range(len(percentageLHSs))]
        EstimatedUpperBounds = [[ 0 for _ in range(len(nDatas))] for _ in range(len(percentageLHSs))]
        for i, percentageLHS in enumerate(percentageLHSs):
            for j, nData in enumerate(nDatas):
                currKeyChoice = (dataSource, nData, round(percentageLHS,2))
                trueValues.append(groupby_object1.get_group(currKeyChoice)['True Value'].unique()[0])
                EstimatedUpperBounds[i][j]= groupby_object1.get_group(currKeyChoice)[targetColumns].mean().tolist()
                RelativeRatios[i][j] = (groupby_object1.get_group(currKeyChoice)[targetColumns].mean().values/trueValues[-1]).tolist()
                CoverageProbabilities[i][j]=(groupby_object1.get_group(currKeyChoice)[targetColumns].values>trueValues[-1]).mean(axis=0).tolist()
        contents.append(tableFiveUnit(percentageLHSs, dataSource, EstimatedUpperBounds, RelativeRatios, CoverageProbabilities))
    ## trueValue should be same among different percentageLHS.         
    assert np.unique(trueValues).size == 1         
    return ("\\hline \n".join(contents), trueValues[0])


def getTableFive(go1:pd.core.groupby.generic.DataFrameGroupBy, 
                 dataSources: list, nDatas: list, percentageLHSs: list, targetColumns: list,
                 title:str, label: str, scalebox: float):
    content, trueValue = tableFive(go1, dataSources, nDatas, percentageLHSs, targetColumns)
    latexTable= r'''
\begin{table}[ht]
    \centering'''+\
    r'''\scalebox{{{:}}}{{'''.format(scalebox)+\
    r'''\begin{tabular}{cc|ccc}
    \toprule
    \hline''' +r'''
    Data Source & LHS Quantitle & Relative Error & Upper Bound & Coverage Probability \\\hline'''+"\n"+\
    content+\
    r'''
    \hline
    \bottomrule
    \end{tabular}}'''+\
    '''\caption{{{:}}}
    \label{{{:}}}
\end{{table}}
    '''.format(title+"The true value is {:}.".format(trueValue), label)
    return latexTable