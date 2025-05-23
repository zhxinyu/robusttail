import numpy as np

def getSignificanceNExponent (value: float):
        exponent = np.floor(np.log(value)/np.log(10))
        return (value/10**exponent, exponent)
    
def tableFiveOneUnit(targetColumns, EstimatedUpperBound, RelativeRatio, CoverageProbability):
    content = r""
    for i, targetColumn in enumerate(targetColumns):
        significance, exponent = getSignificanceNExponent(EstimatedUpperBound[i])
        if targetColumn == '(0,CHI2)':
            content+=r"$(0,\chi^2)$ & ${:.2f}/{:.2f}\times 10^{{{:d}}}$ & ${:.1f}$".format(RelativeRatio[i], significance, int(exponent),CoverageProbability[i])+'''\\\\
        '''
        elif targetColumn == '(1,CHI2)':
            content+=r"$(1,\chi^2)$ & ${:.2f}/{:.2f}\times 10^{{{:d}}}$ & ${:.1f}$".format(RelativeRatio[i], significance, int(exponent),CoverageProbability[i])+'''\\\\
        '''
        elif targetColumn == '(2,CHI2)':
            content+=r"$(2,\chi^2)$ & ${:.2f}/{:.2f}\times 10^{{{:d}}}$ & ${:.1f}$".format(RelativeRatio[i], significance, int(exponent),CoverageProbability[i])+'''\\\\
        '''
        elif targetColumn == '(0,KS)':
            content+=r"$(0,\text{{KS}})$ & ${:.2f}/{:.2f}\times 10^{{{:d}}}$ & ${:.1f}$".format(RelativeRatio[i], significance, int(exponent),CoverageProbability[i])+'''\\\\
        '''
        elif targetColumn == '(1,KS)':
            content+=r"$(1,\text{{KS}})$ & ${:.2f}/{:.2f}\times 10^{{{:d}}}$ & ${:.1f}$".format(RelativeRatio[i], significance, int(exponent),CoverageProbability[i])+'''\\\\
        '''
        elif targetColumn == '(2,KS)':
            content+=r"$(2,\text{{KS}})$ & ${:.2f}/{:.2f}\times 10^{{{:d}}}$ & ${:.1f}$".format(RelativeRatio[i], significance, int(exponent),CoverageProbability[i])+'''\\\\
        '''
        else:
            print(targetColumn)
            assert False
    return content

def tableFiveOneSubTable(content, subtableTitle, subtableLabel, textwidth):
    return r"\begin{subtable}"+r"{{{:}\textwidth}}".format(textwidth)+\
        r'''\begin{tabular}{ccc}
        \toprule
        \hline
        \multicolumn{1}{p{2.8cm}}{Constraint setting} &
        \multicolumn{1}{p{3.9cm}}{Relative ratio/Estimated UpperBound}  &
        \multicolumn{1}{p{2cm}}{Converage probability}  \\\hline
        '''+\
    content+\
    r'''\hline
    \bottomrule
    \end{tabular}
    '''+\
    r'''\caption{{{:}}}
    '''.format(subtableTitle)+\
    r'''\label{{{:}}}
    '''.format(subtableLabel)+\
    r'''\end{subtable}
    '''    
    

def tableFiveOne(groupby_object:pd.core.groupby.generic.DataFrameGroupBy, targetColumns:list,
                 keyChoice1:tuple, keyChoice2:tuple, 
                 subTableTitle1:str, subTableLabel1:str, subTableTitle2:str, subTableLabel2:str,
                 tableTitle:str, tableLabel:str, scalebox:float = 0.7, textwidth:float=0.7):
    trueValue1 = groupby_object.get_group(keyChoice1)['trueValue'].unique()
    RelativeRatio1 = (groupby_object.get_group(keyChoice1)[targetColumns].mean()/trueValue1).values
    EstimatedUpperBound1 = groupby_object.get_group(keyChoice1)[targetColumns].mean().values
    CoverageProbability1 = (groupby_object.get_group(keyChoice1)[targetColumns]>trueValue1[0]).mean().values
    trueValue2 = groupby_object.get_group(keyChoice2)['trueValue'].unique()
    RelativeRatio2 = (groupby_object.get_group(keyChoice2)[targetColumns].mean()/trueValue2).values
    EstimatedUpperBound2 = groupby_object.get_group(keyChoice2)[targetColumns].mean().values
    CoverageProbability2 = (groupby_object.get_group(keyChoice2)[targetColumns]>trueValue2[0]).mean().values
    latexTable = r'''
    \begin{table}[ht]
    \centering'''+\
    r'''\scalebox{{{:}}}{{
    '''.format(scalebox)+\
    tableFiveOneSubTable(
        tableFiveOneUnit(targetColumns, EstimatedUpperBound1, RelativeRatio1, CoverageProbability1),
        subTableTitle1, subTableLabel1, textwidth)+\
    r'''\quad\quad\quad\quad
    '''+\
    tableFiveOneSubTable(
        tableFiveOneUnit(targetColumns, EstimatedUpperBound2, RelativeRatio2, CoverageProbability2),
        subTableTitle2, subTableLabel2, textwidth)+\
    r"}"+\
    r'''\caption{{{:}}}
    '''.format(tableTitle)+\
    r'''\label{{{:}}}
    '''.format(tableLabel)+\
    r'''\end{table}'''
    return latexTable