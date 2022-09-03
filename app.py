# Ian Jenatz y Fabricio Juarez

import numpy as np # manejo de matrices
import sys
import pandas as pd # estructura las matrices para print()
from flask import Flask, render_template, request

'''
Code Description:
This code solves real-life linear programming problems based on the project described by the Final Project Pitch. We use Python Flask to create a simple website running on localhost. The user can then input a maximization problem with two positive variables and n amount of constraints. The program then calculates the optimum with the primal and the dual, and then finds the optimum with integer variables.
'''

def inputData(xvar, yvar, constAmount):
    """
    Description:
        A function that asks for the various inputs required to solve the linear programming problem.
    Args: 
        xvar (any/int): The x value for the objective function.
        yvar (any/int): The y value for the objective function.
        constAmount (any/int): The amount of constraints for the problem.
    Returns:
        varAmount (int): Contains the amount of variables for the problem.
        coefmatrix (numpy array): Matrix containing the coefficients of the objective function in the format x1, x2, ..., xn.
        choiceP (Str): Contains a value (max or min) that dictates if the problem will be maximized or minimized.
        cant_rest (int): Contains the amount of constraints the problem has.
        ursP (array) : contains values 0 or 1 depending on input if it is a urs variable
    """
    varAmount = 2
    coefmatrix = [0.0] * varAmount
    coefmatrix[0] = float(xvar)*-1
    coefmatrix[1] = float(yvar)*-1
    print(coefmatrix)

    choiceP = 'max'

    cant_rest = constAmount
    print(cant_rest, 'restricciones \n')

    ursP = [0] * varAmount

    return varAmount, coefmatrix, choiceP, cant_rest, ursP


def createTable(xval, yval, inequality, total):
    """
    Description:
        A function that generates the initial simplex tableau based on the users' input of data.
    Args: 
        xval (any/int): The x values for the constraints.
        yval (any/int): The y values for the constraints.
        inequality (any/int): The inequalities for the constraints.
        total (any/int): The total values for the constraints.
    Returns:
        tab_matrix (numpy array): Matrix containing the simplex tableau with all variables automatically structured, including slack variables.
        ineq_primal (array): it contains the inequalities symbols
        tableau (dataframe): Structured tab_matrix with pd.
    """
    tab_matrix = np.zeros([int(cant_rest)+1, varAmount + 3], dtype = float)
    ineq_primal = []

    tab_matrix[0][1] = 1 # for z variable
    i = 2
    while i < varAmount + 2: # inputs coefmatrix values into first row.
        tab_matrix[0][i] = coefmatrix[i-2]
        i+=1

    row = 1 # ignores row 0 (z)
    col = 2 # ignores 'row' and 'z' columns.
    while row < int(cant_rest)+1:
        tab_matrix[row][0] = row # generates row # in 'row' column.
        col = 2
        tab_matrix[row][2] = xval[row-1]
        tab_matrix[row][3] = yval[row-1]
        tab_matrix[row][4] = total[row-1]
        ineq_primal.append(inequality[row-1])
        print(tab_matrix)
        print(ineq_primal)
        row+=1

    cols = ['row','z','x1','x2','total']
    print(cols)
    tableau = pd.DataFrame(tab_matrix, columns=cols)
    print(tableau.round(3).to_string(index=False), '\n')

    return tab_matrix, ineq_primal, tableau


def primalDual():
    """
    Description:
        A function that generates the dual matrix based on the primal.
    Args: 
        There are no direct arguments to function.
    Returns:
        tableau_iniciald (dataframe): Structured dual_matrix with pd.
    """
    global slackAmountP, varAmountP, cant_restP, dual_matrix, slackAmountD, varAmountD, cant_restD, choiceD, ineq_dual, ursD
    varAmountP = len(primal_matrix[0]) - 3
    print(varAmountP)
    cant_restP = len(primal_matrix) - 1
    slackAmountP = 0

    for i in ineq_primal:
        if i != '=':
            slackAmountP += 1

    dual_matrix = primal_matrix.copy()
    dual_matrix = np.delete(dual_matrix, [0, 1], 1)

    choiceD = 'min'
    if choiceP == 'min':
        choiceD = 'max'

    normal = 1
    if choiceP == 'max':
        for i in ineq_primal:
            if i != '<=':
                normal = 0
    if choiceP == 'min':
        for i in ineq_primal:
            if i != '>=':
                normal = 0

    ursD = [0] * cant_restP
    ineq_dual = []
    if normal == 1:
        if choiceD == 'max':
            for i in range(varAmountP):
                ineq_dual.append('<=')
        elif choiceD == 'min':
            for i in range(varAmountP):
                ineq_dual.append('>=')
    elif normal == 0:
        if choiceP == 'max':
            i = 0
            while i < len(ineq_primal):
                if ineq_primal[i] == '>=':
                    dual_matrix[i+1] *= -1
                elif ineq_primal[i] == '=':
                    ursD[i] = 1
                ineq_dual.append('>=')
                if i < len(ursP):
                    if ursP[i] == 1:
                        ineq_dual.append('=')
                i += 1
        elif choiceP == 'min':
            i = 0
            while i < len(ineq_primal):
                if ineq_primal[i] == '<=':
                    dual_matrix[i+1] *= -1
                elif ineq_primal[i] == '=':
                    ursD[i] = 1
                ineq_dual.append('<=')
                if i < len(ursP):
                    if ursP[i] == 1:
                        ineq_dual.append('=')
                i += 1

    while len(ineq_dual) > varAmountP:
        ineq_dual.pop()

    while len(ineq_dual) < varAmountP:
        if choiceP == 'max':
            ineq_dual.append('>=')
        elif choiceP == 'min':
            ineq_dual.append('<=')

    dual_matrix = np.roll(dual_matrix,-1,0)
    dual_matrix = dual_matrix.T
    dual_matrix = np.roll(dual_matrix,1,0)

    dual_matrix = np.insert(dual_matrix, 0, 0, 1)
    dual_matrix = np.insert(dual_matrix, 0, 0, 1)
    dual_matrix[0][1] = 1
    i = 0
    while i < len(dual_matrix):
        dual_matrix[i][0] = i
        i += 1

    i = 1
    while i < len(dual_matrix):
        dual_matrix[i][-1] *= -1
        i += 1

    i = 2
    while i < len(dual_matrix[0]) - 1:
        dual_matrix[0][i] *= -1
        i += 1

    print("Primal Matrix")
    print(primal_matrix[:,2:])
    print(ineq_primal)
    print(ursP)
    print("Dual Matrix")
    print(dual_matrix[:,2:])
    print(ineq_dual)
    print(ursD)

    varAmountD = cant_restP
    cant_restD = varAmountP
    slackAmountD = 0
    for i in ineq_dual:
        if i != '=':
            slackAmountD += 1

    cols = ['row','z']
    i = 0
    while i < int(cant_rest):
        tempo = str(i + 1)
        cols.append(f"x{tempo}")
        i+=1
    cols.append('total')
    print(cols)
    tableau_iniciald = pd.DataFrame(dual_matrix, columns=cols)
    print (cant_rest)

    return tableau_iniciald


def initialTable(tab_matrix, slackAmount, varAmount, cant_rest, choice, inequalities, urs):
    """
    Description:
    A function that makes the neccesary adjustments to the matrix in order to be ready to be solved
    Args: 
        tab matrix (numpy array): containing the corresponding information of the problem
        slack amount (int): number of slack variables neccessary
        var amount (int): number of variables the problem has
        cant rest (int): number of restrictions the problem has
        choice (string): information of wether the problem is a max or min problem
        inequalities(array): contains the inequalities symbols of the problem
        urs (array): contains the urs information
    Returns:
        tab_matrix (numpy array): Matrix containing the simplex tableau with all variables automatically structured, including slack variables.
        cols (array): returns name of all the columns 
        choice (string): information of wether the problem is a max or min problem
        varAmount (int): returns the number of variables
    """
    # Primal Big M
    print(inequalities)
    i = 1
    while i < len(tab_matrix): # Checks total in each row and if negative, multiplies row by -1 and changes inequality symbol.
        if tab_matrix[i][-1] < 0:
            print('yeah')
            j = 2
            while j < len(tab_matrix[0] + 1):
                tab_matrix[i][j] *= -1
                j += 1
            if inequalities[i-1] == '<=':
                print("no")
                inequalities[i-1] = '>='
            elif inequalities[i-1] == '>=':
                print("yup")
                inequalities[i-1] = '<=' 
        i += 1
    print(inequalities)

    for i in range(slackAmount):
        tab_matrix = np.insert(tab_matrix, 2 + varAmount, 0, 1)

    print(tab_matrix)

    print(choice)
    row = 1
    actRow = 1
    col = 2 + varAmount
    artifAmount = 0
    if choice == 'max':
        mVal = 1000000000
    elif choice == 'min':
        mVal = -1000000000
    artifCols = []
    print(mVal)

    while row < len(tab_matrix):
        if inequalities[row - 1] == '<=':
            tab_matrix[row][col + actRow-1] = 1
        elif inequalities[row - 1] == '>=':
            tab_matrix[row][col + actRow-1] = -1
            tab_matrix = np.insert(tab_matrix, len(tab_matrix[0]) - 1, 0, 1)
            tab_matrix[row][len(tab_matrix[0]) - 2] = 1
            tab_matrix[0][len(tab_matrix[0]) - 2] = mVal
            artifCols.append(row)
            artifAmount += 1
        elif inequalities[row - 1] == '=':
            tab_matrix = np.insert(tab_matrix, len(tab_matrix[0]) - 1, 0, 1)
            tab_matrix[row][len(tab_matrix[0]) - 2] = 1
            tab_matrix[0][len(tab_matrix[0]) - 2] = mVal
            artifCols.append(row)
            artifAmount += 1
            actRow -= 1
        row += 1
        actRow += 1
    print(tab_matrix)

    print(urs)
    i = 0
    while i < len(urs):
        if urs[i] == 1:
            tab_matrix = np.insert(tab_matrix, len(tab_matrix[0]) - 1, 0, 1)
            row = 0
            while row < len(tab_matrix):
                tab_matrix[row][-2] = tab_matrix[row][i + 2] * -1
                row += 1
        i += 1

    col = 2
    while col < len(tab_matrix[0]):
        row = 1
        while row < len(tab_matrix):
            if row in artifCols:
                tab_matrix[0][col] += (tab_matrix[row][col] * mVal * -1)
            row += 1
        col += 1

    print("\nInitial Tableau: \n")
    cols = ["Row", "z"]
    i = 0
    while i < varAmount:
        cols.append(f"x{i+1}")
        i+=1
    j = 0
    while j < cant_rest:
        if inequalities[j] == "<=":
            cols.append(f"s{j+1}")
        elif inequalities[j] == ">=":
            cols.append(f"e{j+1}")
        j+=1
    k = 0
    while k < cant_rest:
        if inequalities[k] == ">=":
            cols.append(f"a{k+1}")
        elif inequalities[k] == "=":
            cols.append(f"a{k+1}")
        k+=1
    l = 0
    while l < len(urs):
        if urs[l] == 1:
            cols.append(f"x{l+1}''")
        l += 1

    cols.append("Total")
    print(cols)
    tableau = pd.DataFrame(tab_matrix, columns=cols)
    print(tableau.round(3).to_string(index=False), '\n')

    return tab_matrix, cols, choice, varAmount,tableau


# Simplex (Big M)
def findValues():
    """
    Description:
        A function that finds the important values to be used to simplify the table according to the simplex method.
    Args: 
        There are no direct arguments to function.
    Returns:
        divVal (float): Contains the value that the next iteration's pivot row will be divided by.
        min_index (int) Contains the index of the divVal's row.
        min_val (float): Contains the minimum total amount in the table, calculated by dividing each rows total amount by its value in the column of the pivot.
    """
    print(tab_matrix)
    if choice == 'min':
        min_val_arr = np.where(tab_matrix[0] == np.amax(tab_matrix[0][2:len(tab_matrix[0]) - 1]))
    elif choice == 'max':
        min_val_arr = np.where(tab_matrix[0] == np.amin(tab_matrix[0][2:len(tab_matrix[0]) - 1]))
    
    i = 0
    while i < len(min_val_arr[0]):
        if min_val_arr[0][i] == 0 or min_val_arr[0][i] == 1:
            min_val_arr = np.delete(min_val_arr, i, 1)
        i += 1
    print(min_val_arr)
    try:
        min_val = int(min_val_arr[0])
    except:
        min_val = int(min_val_arr[0][0])
    pivot = tab_matrix[0][min_val]
    print("Column: ", min_val)
    print("Pivot: ", pivot)

    min_total = float('inf')
    row = 1
    while row < len(tab_matrix):
        if tab_matrix[row][min_val] > 0 and tab_matrix[row][-1]/tab_matrix[row][min_val] < min_total and tab_matrix[row][-1]/tab_matrix[row][min_val] >= 0:
            min_total = tab_matrix[row][-1]/tab_matrix[row][min_val]
            divVal = tab_matrix[row][min_val]
            min_index = row 
        row+=1

    print("Minimum total: ", min_total)
    if min_total == float('inf'):
        sys.exit("No solution")
    print("Value to divide by: ", divVal)
    print("row: ", min_index, "\n")

    return divVal, min_index, min_val


def changeTable():
    """
    Description:
        A function that modifies the simplex table utilizing each row's corresponding formula.
    Args: 
        There are no direct arguments to function.
    Returns:
        There are no returns for the function since the variables are global. This function also only modifies the tab_matrix (numpy array) variable which is the simplex tableau.
    """
    matrix_copy = tab_matrix.copy()
    col = 2
    while col < len(tab_matrix[min_index]):
        tab_matrix[min_index][col] = matrix_copy[min_index][col] / divVal
        col += 1

    row = 0
    col = 1
    while row < len(tab_matrix):
        col = 1
        if row != min_index:
            while col < len(tab_matrix[0]):
                tab_matrix[row][col] = matrix_copy[row][col] - (matrix_copy[row][min_val] * tab_matrix[min_index][col])
                col += 1
        row += 1

    tableau = pd.DataFrame(tab_matrix, columns=cols)
    print(tableau.round(3).to_string(index=False), "\n")


def checkIteration():
    """
    Description:
        A function that checks the first row in the simplex table and changes value depending on if another iteration is necessary or not.
    Args: 
        There are no direct arguments to function.
    Returns:
        iteration (bool) A True or False value that dictates if another iteration is necessary.
    """
    iteration = False
    if choice == 'min':
        for i in tab_matrix[0][2:len(tab_matrix[0]) - 1]:
            if i > 0:
                iteration = True
    if choice == 'max':
        for i in tab_matrix[0][2:len(tab_matrix[0]) - 1]:
            if i < 0:
                iteration = True
    return iteration



# Find best solution for integer problems
# Find all possible points.
def findMaxVals():
    """
    Description:
        Finds the max values for both the x and y values, these values become the possible range for these variables.
    Args: 
        There are no direct arguments to function.
    Returns:
        xVal (any/float) The max value of the x variable.
        yVal (any/float) The max value of the y variable.
    """
    mylist, maxXVals, maxYVals = [], [], []
    for i in constraint_matrix[1:]:
        for j in i[2:-1]:
            mylist.append(j)

    xVals = [int(mylist[i]) for i in range(len(mylist)) if i % 2 == 0]
    yVals = [int(mylist[i]) for i in range(len(mylist)) if i % 2 != 0]
    print(xVals, yVals, sep='\n')
    
    i = 1
    while i < len(constraint_matrix):
        if constraint_matrix[i][2] != 0:
            resultX = constraint_matrix[i][-1] / constraint_matrix[i][2]
            if resultX != float('inf'):
                maxXVals.append(resultX)
        if constraint_matrix[i][3] != 0:
            resultY = constraint_matrix[i][-1] / constraint_matrix[i][3]
            if resultY != float('inf'):
                maxYVals.append(resultY)
        i += 1
    print(maxXVals)
    print(maxYVals)

    if '>=' in inequalities:
        xVal = max(maxXVals)
        yVal = max(maxYVals)
    else:
        xVal = min(maxXVals)
        yVal = min(maxYVals)
    
    print(xVal, yVal, sep='\n')
    return xVal, yVal


def getallPoints():
    """
    Description:
        Finds all possible points for the problem. 
    Args: 
        There are no direct arguments to function.
    Returns:
        allPoints (list) A list of all possible points for the problem (without checking if they are possible in the constraints).
    """
    allPoints = []
    for x in range(0, round(xVal)+1):
        for y in range(0, round(yVal)+1):
            allPoints.append((x, y))
    return allPoints


def getRealP():
    """
    Description:
        Finds all feasible points for the problem by checking each point within the problem's constraints. 
    Args: 
        There are no direct arguments to function.
    Returns:
        real (list) A list of all feasible points for the problem.
    """
    real = all_points.copy()
    i = 0
    while i < len(all_points):
        amount = 0
        j = 0
        while j < len(constraint_matrix[1:]):
            amount = all_points[i][0] * constraint_matrix[j+1][2] + all_points[i][1] * constraint_matrix[j+1][3]

            if (inequalities[j] == '<='):
                if (amount > constraint_matrix[j+1][-1]):
                    real[i] = 0
            elif (inequalities[j] == '>='):
                if (amount < constraint_matrix[j+1][-1]):
                    real[i] = 0

            j+=1
        i+=1
    
    r = []
    for i in real:
        try:
            int(i)
        except:
            r.append(i)

    real = r
    return real

# print("\nFeasible POI")
# print(real)


def Max(mylist):
    """
    Description:
        Finds the maximum value between the results of the Objective Function.
    Args:
        mylist (list): List of all the results of the objective function.
    Return:
        (Str): Maximum value and the point x, y that generates this maximum value.
    """
    i = 0
    val = mylist[0]
    point = real[0]
    while i < len(mylist):
        if mylist[i] > val:
            val = mylist[i]
            point = real[i]
        i+=1
    return "Max Value is " + str(int(val)) + ", Point of Interest is " + str(point)


def getResults(real):
    """
    Description:
        Receives an array of all the possible optimal points and it runs them in the FO and then returns the maximum value and the x, y point for this value.
    Args:
        real (list): List of all the possible optimal points
    Return:
        optimal (list): Maximum value and the point x, y that generates this maximum value.
    """
    milista = []
    index = 0
    while index < len(real):
        total_amount = real[index][0]*(constraint_matrix[0][2]*-1) + real[index][1]*(constraint_matrix[0][3]*-1)
        milista.append(total_amount)
        index+=1

    optimal = Max(milista)
    return optimal



app = Flask(__name__)

@app.route('/inputs')
def inputs():
    """
    Description:
        HTML webpage on endpoint '/inputs' that allows users to enter the input variables (Objective Function x and y values and the amount of constraints).
    Args:
        There are no direct arguments to function.
    Return:
        (html str): HTML webpage where the user can input the input variables for the problem
    """
    return render_template('inputs.html', consts = [])


@app.route('/constraints', methods=['POST'])
def constraints():
    """
    Description:
        POST method HTML webpage on endpoint '/constraints' that allows users to change the input variables or enter the constraints for the problem. It receives the values from the input variable form and calls the inputData method.
    Args:
        There are no direct arguments to function.
    Return:
        (html str): HTML webpage where the user can modify the input variables for the problem or input the constraints.
    """
    xvar = request.form['xvar']
    yvar = request.form['yvar']
    constAmount = request.form['constAmount']

    global varAmount, coefmatrix, choiceP, cant_rest, ursP
    varAmount, coefmatrix, choiceP, cant_rest, ursP = inputData(xvar, yvar, constAmount)
    print(xvar, yvar)
    consts = [1]*int(cant_rest)
    return render_template('inputs.html', consts = consts)


@app.route('/output', methods=['POST'])
def output():
    """
    Description:
        POST method HTML webpage on endpoint '/output' that returns all relevant outputs for the linear programming problem. It receives the values from the input constraints form and calls various methods.
    Args:
        There are no direct arguments to function.
    Return:
        (html str): HTML webpage that prints all relevant outputs for the linear programming problem.
    """
    xval = request.form.getlist('xval')
    yval = request.form.getlist('yval')
    inequality = request.form.getlist('inequality')
    total = request.form.getlist('total')

    global primal_matrix, ineq_primal
    primal_matrix, ineq_primal,tableau_inicialp = createTable(xval, yval, inequality, total)
    
    # Big M


    tableau_iniciald = primalDual()
    
    global constraint_matrix, inequalities, tab_matrix, cols, choice, varAmount, divVal, min_index, min_val, tab_matrix_dual, cols_dual, choice_dual, varAmount_dual, divVal_dual, min_index_dual, min_val_dual
    constraint_matrix = primal_matrix
    inequalities = ineq_primal
    tab_matrix, cols, choice, varAmount,tableu_finalp = initialTable(primal_matrix, slackAmountP, varAmountP, cant_restP, choiceP, ineq_primal, ursP)
    iteration = checkIteration()
    while iteration == True:
        divVal, min_index, min_val = findValues()
        changeTable()
        iteration = checkIteration()

    print("\nOptimal Solution Primal: ", tab_matrix[0][-1], "\n\n\n\n\n\n")

    tab_matrix_dual, cols_dual, choice_dual, varAmount_dual,tableu_finald = initialTable(dual_matrix, slackAmountD, varAmountD, cant_restD, choiceD, ineq_dual, ursD)
    iteration = checkIteration()
    while iteration == True:
        divVal_dual, min_index_dual, min_val_dual = findValues()
        changeTable()
        iteration = checkIteration()

    print("\nOptimal Solution Dual: ", tab_matrix[0][-1])


    # Integer Max


    global xVal, yVal, all_points, real
    xVal, yVal = findMaxVals()

    all_points = getallPoints()

    real = getRealP()

    solutionInt = getResults(real)
    print("\nTotals: ")
    print(solutionInt)
    
    
    return render_template('output.html',
        varAmountP = varAmountP,
        cant_restP = cant_restP,
        choiceP = choiceP,
        ineq_primal = ineq_primal,
        tablespi =  [tableau_inicialp.to_html(classes='data', header="true")],

        varAmountD = varAmountD,
        cant_restD = cant_restD,
        choiceD = choiceD,
        ineq_dual = ineq_dual,
        tablesdi = [tableau_iniciald.to_html(classes='data', header="true")],

        tablesp =[tableu_finalp.to_html(classes='data', header="true")],
        tablesd = [tableu_finald.to_html(classes='data', header="true")] ,
        solution = tab_matrix[0][-1],

        xVal = xVal,
        yVal = yVal,
        solutionInt = solutionInt
    )

if __name__ == '__main__':
    app.run(debug = True)


# # Instrucciones Ejercicio
# primal_matrix = np.array(
# [[ 0.0,  1.0,  -10.0,  -15.0,  0.0],
# [ 1.0,  0.0,  282.0,  400.0,  2000.0],
# [ 2.0,  0.0,  4.0,  40.0,  140.0],
# [ 3.0,  0.0,  1.0,  0.0,  5.0]])
# choiceP = 'max'
# ineq_primal = ['<=', '<=', '<=']
# ursP = [0, 0]
# Respuesta = 70, (4, 2)

# # Ejercicio Prueba PF 1
# primal_matrix = np.array(
# [[ 0.0,  1.0,  -5.0,  -9.0,  0.0],
# [ 1.0,  0.0,  3.0,  5.0,  60.0],
# [ 2.0,  0.0,  4.0,  4.0,  72.0],
# [ 3.0,  0.0,  2.0,  4.0,  100.0]])
# choiceP = 'max'
# ineq_primal = ['<=', '<=', '<=']
# ursP = [0, 0]
# Respuesta = 108, (0, 12)

# # Ejercicio Prueba PF 2
# primal_matrix = np.array(
# [[ 0.0,  1.0,  -4.0,  -3.0,  0.0],
# [ 1.0,  0.0,  2.0,  1.0,  1000.0],
# [ 2.0,  0.0,  1.0,  1.0,  800.0],
# [ 3.0,  0.0,  1.0,  0.0,  400.0],
# [ 3.0,  0.0,  0.0,  1.0,  700.0]])
# choiceP = 'max'
# ineq_primal = ['<=', '<=', '<=', '<=']
# ursP = [0, 0]
# Respuesta = 2600, (200, 600)

# # Ejercicio Prueba PF 3
# primal_matrix = np.array(
# [[ 0.0,  1.0,  -3.0,  -9.0,  0.0],
# [ 1.0,  0.0,  1.0,  4.0,  8.0],
# [ 2.0,  0.0,  1.0,  2.0,  4.0]])
# choiceP = 'max'
# ineq_primal = ['<=', '<=']
# ursP = [0, 0]
# Respuesta = 18, (0, 2)

# # Ejercicio Prueba PF 4
# primal_matrix = np.array(
# [[ 0.0,  1.0,  -8.0,  -4.0,  0.0],
# [ 1.0,  0.0,  2.0,  3.0,  30.0],
# [ 2.0,  0.0,  3.0,  2.0,  24.0],
# [ 3.0,  0.0,  1.0,  1.0,  6.0]])
# choiceP = 'max'
# ineq_primal = ['<=', '<=', '>=']
# ursP = [0, 0]
# Respuesta = 64, (8, 0)
