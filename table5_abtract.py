from table5_central_import import *

def runner(args):
    randomSeed = 20220222
    stringToDataModule = {"gamma": gamma,
                          "lognorm": lognorm,
                          "pareto": pareto}
    trueValue = 0.005
    metaDataDict = {"dataSize": 500}
    percentageLHSs = [args.lhs]
    dataSources = [ args.ds ]
    method = args.method
    if method == 'pot':
        with open('gpdTIP_pot.R','r') as f:
            RCodeLib = f.read()    
    elif method == 'pot_bt':
        with open('gpdTIP_pot_bt.R','r') as f:
            RCodeLib = f.read()
    elif method == 'pl':
        with open('gpdTIP_pl.R','r') as f:
            RCodeLib = f.read()
    elif method == 'bayesian':
        with open('gpdTIP_bayesian.R','r') as f:
            RCodeLib = f.read()
    elif method == 'pwm':
        with open('gpdTIP_pwm.R','r') as f:
            RCodeLib = f.read()
    else:
        raise NotImplementedError()
    
    file_dir = f'large_table5_{method}'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    nrep = 200
    result = []
    columns = ["Data Source", 'nData',"percentageLHS", "Lower Bound","Upper Bound", "True Value", "Repetition Index"]
    for percentageLHS in percentageLHSs:
        percentageLHS = np.round(percentageLHS, 2)
        percentageRHS = percentageLHS + trueValue
        for dataSource in dataSources:
            dataModule = stringToDataModule[dataSource]
            leftEndPointObjective = dpu.endPointGeneration(
                dataModule, percentageLHS, dpu.dataModuleToDefaultParamDict[dataModule])
            rightEndPointObjective = dpu.endPointGeneration(
                dataModule, percentageRHS, dpu.dataModuleToDefaultParamDict[dataModule])
            for nnrep in range(nrep):
                print(f"Working on {percentageLHS}_{dataSource}_{nnrep}")
                try:
                    metaDataDict['random_state'] = randomSeed+nnrep
                    RCodeApply = f'''
lhs <- {leftEndPointObjective}

rhs <- {rightEndPointObjective}

data <-  c(t(read.csv("./large_data/{dataSource}/default/randomseed={randomSeed+nnrep}.csv", header=FALSE)))

out <- tryCatch(
	gpdTIP(data, lhs, rhs, conf=0.95), 
	error = function(e) e
)

if ("error" %in% class(out)) {{
  print(out)
}}

bbd <- if ("error" %in% class(out)) NA else{{
	out$CI
}}
bbd 
                    '''
                    roResult = ro.r(RCodeLib+RCodeApply)
                    result.append([dataSource, metaDataDict['dataSize'], percentageLHS, roResult[0], roResult[1], trueValue, nnrep])
                    print(result[-1])
                except Exception as e: 
                    print(e)
                    result.append([dataSource, metaDataDict['dataSize'], 
                                percentageLHS, 0, 
                                0, 
                                trueValue, nnrep])
            # assert False
            print(f"Finish experiments on {percentageLHS}-{dataSource}")
            df = pd.DataFrame(data = result, columns = columns)
            df.to_csv(os.path.join(file_dir, f'table5_{percentageLHS}_{dataSource}.csv'), header=columns, index = False)

if __name__ == '__main__':

    """Usage
    
        ```
        #!/bin/bash 
        for lhs in 0.9 0.95 0.99
        do
            echo "Running" ${lhs} gamma pl
            python table5_ab.py ${lhs} gamma pl &
            echo "Running" ${lhs} lognorm pl
            python table5_ab.py ${lhs} lognorm pl &
            echo "Running" ${lhs} pareto pl
            python table5_ab.py ${lhs} pareto pl &
        done
        wait 
        echo "done all processes."
        ``` 
    """
    parser = argparse.ArgumentParser(description='TIP estimation with user-specific methods.')
    parser.add_argument('lhs', type=float, help='LHS in the objective function')
    parser.add_argument('ds', type=str, help='Data source for simulation')
    parser.add_argument('method', type=str, help='choose the method for tail probability estimation: pot, pl, bayesian and pwm')
    args = parser.parse_args()
    
    runner(args)