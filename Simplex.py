import numpy as np
import csv
import sys


def main():
    # read csv files and check the dimention of input
    l = readInput() 

    solve(*l)

def solve(A, b, c):
    with open("solution.tex", "w") as f:
        f.write(r'''\documentclass [12pt] {article}
\usepackage{amsmath}
\usepackage{url}
\pagestyle{plain}
\begin{document}''')


    B_set, N_set = [], []
    for i in range(c.shape[1]):
        N_set.append(i)
    for j in range(b.shape[1]):
        B_set.append(j + c.shape[1])
    N = A
    B = np.identity(b.shape[1])
    A_conca = np.concatenate((N, B))
    x_B_star = b
    z_N_star = 0 - c

    if any(val < 0 for val in x_B_star.tolist()[0]) and any(val > 0 for val in (0 - z_N_star).tolist()[0]):
        #
        # print("We need dual-based Phase 1 initialization!\n")
        # exit(0)
        #
        # dual-based phase 1
        # change the objective function of the primal to make it dual feasible
        z_N_star_p = np.zeros(len(N_set))
        for i in range(len(N_set)):
            z_N_star_p[i] = 1
        z_N_star_p = np.matrix(z_N_star_p)
        # now start the dual phase 1
        phaTwo = dualSimplex(
            N_set, B_set, N, B, z_N_star_p, x_B_star, A_conca, 1)
        # phase 2
        # recover zN* (c_N) of the primal based on new bases
        N_set = phaTwo[0]
        B_set = phaTwo[1]

        N = phaTwo[2]
        B = phaTwo[3]
        z_N_star = phaTwo[4]
        x_B_star = phaTwo[5]
        A_conca = phaTwo[6]
        z_phase1 = phaTwo[7]
        negTrans = 0 - np.transpose(np.dot(N, np.linalg.inv(B)))
        z_N_star_p = np.matrix(np.zeros(len(N_set)))
        for i in range(z_N_star.shape[1]):
            if i in B_set:
                z_N_star_p += negTrans[B_set.index(i), :]
            if i in N_set:
                tmp = np.zeros(len(N_set))
                tmp[i] = z_N_star[0, N_set.index(i)]
                z_N_star_p -= np.matrix(tmp)
        for i in range(c.shape[1]):
            if c[0, i] == 0:
                z_N_star_p[0, i] = 0
        # now compute the primal phase 2 based upon the optimal solution from
        # phase 1
        z = primalSimplex(N_set, B_set, N, B, z_N_star_p, x_B_star, A_conca)
        # sum up the objective function values of two phases
        if z != "unbounded":
            with open("solution.tex", "a") as f:
                f.write("\nThe optimal objective function is %.2f + %.2f = %.2f" % (z_phase1, z, z_phase1 + z))
            z += z_phase1

    elif all(val >= 0 for val in x_B_star.tolist()[0]) and any(val > 0 for val in (0 - z_N_star).tolist()[0]):
        z = primalSimplex(N_set, B_set, N, B, z_N_star, x_B_star, A_conca)
    elif any(val < 0 for val in x_B_star.tolist()[0]) and all(val <= 0 for val in (0 - z_N_star).tolist()[0]):
        z = dualSimplex(N_set, B_set, N, B, z_N_star, x_B_star, A_conca, 0)
    else:
        z = "optimal"
    if z == "optimal" or z == "unbounded":
        with open("solution.tex", "a") as f:
            f.write("The system is \\underline{\\textbf{%s}}" % z)

    with open("solution.tex", "a") as f:
        f.write(r'''
\end{document}''')
        
    return z


def dualSimplex(N_set, B_set, N, B, z_N_star, x_B_star, A_conca, flag):
    c = 0 - z_N_star
    A = N
    b = x_B_star
    n = len(N_set)
    #
    #   print the original dictionary
    #
    content = r'''\section{Input}\begin{alignat*}{3}min\quad%(obj)s\\s.t.\quad%(cstr)s\end{alignat*}'''

    x = []
    for i in range(1, c.shape[1] + 1):
        x.append("x_"+str(i))

    dic = {}

    obj = ""
    i = 0
    for val in np.nditer(c):
        obj += "%.2f%s" % (val, x[i])
        if (i + 1 < c.shape[1]):
            i += 1
            obj += "&+"
    dic["obj"] = obj

    cstr = ""
    A_list = A.tolist()
    b_list = b.tolist()
    l_j = len(A_list)
    l_i = len(A_list[0])
    i, j = 0, 0
    while 1:
        cstr += "%.2f%s" % (A_list[j][i], x[j])
        if (j + 1 < l_j):
            j += 1
            cstr += "&+"
        else:
            cstr += "&&\\geq%.2f" % b_list[0][i]
            i += 1
            if (i >= l_i):
                break
            cstr += "\\\\"
            j = (j + 1) % l_j
    cstr += ""
    dic["cstr"] = cstr
    with open("solution.tex", "a") as f:
        f.write(content % dic)
    #
    #   write the primal simplex
    #
    dic = {}

    with open("solution.tex", "a") as f:
        f.write("\n\\section{Calculation}\nThe matrix A is given by")

    content = r'''\[\begin{bmatrix}%(matrix)s\end{bmatrix}\]'''

    dic["matrix"] = texMatrix(A_conca)
    with open("solution.tex", "a") as f:
        f.write(content % dic)

    content = r'''
\\The initial sets of basic and non-basic indices are
\[\mathcal{B}=\{%(bsc)s\}\]
\[\mathcal{N}=\{%(nbsc)s\}\]
Corresponding to these sets, we have the submatrices of A:
\[B=\begin{bmatrix}%(bscm)s\end{bmatrix}\]
\[N=\begin{bmatrix}%(nbscm)s\end{bmatrix}\]
'''

    bsc = ""
    for i in range(len(B_set)):
        bsc += str(B_set[i])
        if i != len(B_set) - 1:
            bsc += ","
    dic["bsc"] = bsc

    nbsc = ""
    for i in range(len(N_set)):
        nbsc += str(N_set[i])
        if i != len(N_set) - 1:
            nbsc += ","
    dic["nbsc"] = nbsc

    dic["bscm"] = texMatrix(B)
    dic["nbscm"] = texMatrix(N)

    with open("solution.tex", "a") as f:
        f.write(content % dic)

    iteration = 0
    # now we enter the loop
    while True:
        # find the min in xB*
        m = 0
        for val in x_B_star.tolist()[0]:
            if val < m:
                m = val
        # if no negative in xB*, then optimal
        if m >= 0:
            # store the solution into a dict, key being the index, data being
            # the value
            content = r'''\subsection{Optimal solution:}\[%(solu)s\]\[%(optobj)s\]'''
            sol = {}
            solu = ''
            sol = {}
            for i in range(len(N_set)):
                sol[N_set[i]] = z_N_star.tolist()[0][i]
            for j in range(len(B_set)):
                sol[B_set[j]] = x_B_star.tolist()[0][j]
            # print out solution, N.B. first then B.
            for key in range(A_conca.shape[0]):
                if key < n:
                    solu += "y_%i = %.2f  " % (key + 1, sol[key])
                else:
                    solu += "z_%i = %.2f  " % (key + 1 - n, sol[key])
                if key != A_conca.shape[0] - 1:
                    solu += "\\quad "
            dic["solu"] = solu
            # compute c_B
            c_B = []
            for i in B_set:
                if i < n:
                    c_B.append(c[0, i])
                else:
                    c_B.append(0)
            # compute the objective
            z = np.dot(x_B_star, np.transpose(c_B)).tolist()[0][0]
            optobj = ''
            for i in range(len(c_B)):
                optobj += "%.2f * %.2f + " % (c_B[i], sol[B_set[i]])
                if i != len(c_B) - 1:
                    optobj += '+ '
            optobj += "= %.2f\n" % z
            dic["optobj"] = optobj
            with open("solution.tex", "a") as f:
                f.write(content % dic)
            if (flag):
                return [N_set, B_set, N, B, z_N_star, x_B_star, A_conca, z]
            return z

        content = r'''
\subsection{Iteration %(iter)s:}
\[i = %(i)s\]
\[e_{i} = \begin{bmatrix}%(e_i)s\end{bmatrix}\]
\[\Delta Z_{\mathcal N} = -(B^{-1} N)^{T} e_i = -(\begin{bmatrix}%(Binv)s\end{bmatrix} \begin{bmatrix}%(N)s\end{bmatrix})^{T} \begin{bmatrix}%(e_i)s\end{bmatrix} = \begin{bmatrix}%(del_z_N)s\end{bmatrix}\]
\[s = (max\{%(div)s\})^{-1}= %(s)s\]
\[j = %(j)s\]
\[e_{j} = \begin{bmatrix}%(e_j)s\end{bmatrix}\]
\[\Delta X_{\mathcal B} = B^{-1} N e_j = \begin{bmatrix}%(Binv)s\end{bmatrix} \begin{bmatrix}%(N)s\end{bmatrix} \begin{bmatrix}%(e_j)s\end{bmatrix} = \begin{bmatrix}%(del_x_B)s\end{bmatrix}\]
\[t = \frac{X_{i}^{*}}{\Delta X_{i}}= %(t)s\]
\[X_{\mathcal B}^{*} = X_{\mathcal B}^{*} - t \Delta X_{\mathcal B} + t * e_i = \begin{bmatrix}%(xB*)s\end{bmatrix}\]
\[Z_{\mathcal N}^{*} = Z_{\mathcal N}^{*} - s \Delta Z_{\mathcal N} + s * e_j = \begin{bmatrix}%(zN*)s\end{bmatrix}\]
\[\mathcal{B}=\{%(bsc)s\}\]
\[\mathcal{N}=\{%(nbsc)s\}\]
\[B=\begin{bmatrix}%(bscm)s\end{bmatrix}N=\begin{bmatrix}%(nbscm)s\end{bmatrix}\]
'''
        dic["iter"] = iteration + 1
        dic["Binv"] = texMatrix(np.linalg.inv(B))
        dic["N"] = texMatrix(N)
        # entering index
        i = x_B_star.tolist()[0].index(m)
        dic["i"] = i
        # construct e_i
        e_i = np.zeros(len(B_set))
        e_i[i] = 1
        e_i = np.matrix(e_i)
        dic["e_i"] = texMatrix(e_i)
        # compute del_z_N
        del_z_N = 0 - np.dot(e_i, np.transpose(np.dot(N, np.linalg.inv(B))))
        dic["del_z_N"] = texMatrix(del_z_N)
        # compute s and leaving index j
        v_s = []
        div = ''
        for j in range(z_N_star.shape[1]):
            if z_N_star[0, j] == 0:
                # convention for 0/0
                if del_z_N[0, j] == 0:
                    v_s.append(0)
                # +/- inf
                else:
                    v_s.append(del_z_N[0, i] > 0 and np.inf or 0 - np.inf)
            else:
                v_s.append(del_z_N[0, j] / z_N_star[0, j])
            div += '\\frac{%.1f}{%.1f}' % (del_z_N[0, j], z_N_star[0, j])
            if i != z_N_star.shape[1] - 1:
                div += ','
        dic["div"] = div
        s = max(v_s)
        if s <= 0:
            return "unbounded"
        j = v_s.index(s)
        dic["j"] = j
        s = 1 / s
        dic["s"] = s
        # construct e_j
        e_j = np.zeros(len(N_set))
        e_j[j] = 1
        e_j = np.matrix(e_j)
        dic["e_j"] = texMatrix(e_j)
        # compute del_x_B
        del_x_B = np.dot(e_j, np.dot(N, np.linalg.inv(B)))
        dic["del_x_B"] = texMatrix(del_x_B)
        # compute t
        t = x_B_star[0, i] / del_x_B[0, i]
        dic["t"] = t
        # update xB* and zN*
        x_B_star = x_B_star - t * del_x_B
        x_B_star[0, i] = t
        dic["xB*"] = texMatrix(x_B_star)
        z_N_star = z_N_star - s * del_z_N
        z_N_star[0, j] = s
        dic["zN*"] = texMatrix(z_N_star)
        # update B and N set
        tmp = B_set[i]
        B_set[i] = N_set[j]
        N_set[j] = tmp

        bsc = ""
        for i in range(len(B_set)):
            bsc += str(B_set[i])
            if i != len(B_set) - 1:
                bsc += ","
        dic["bsc"] = bsc

        nbsc = ""
        for i in range(len(N_set)):
            nbsc += str(N_set[i])
            if i != len(N_set) - 1:
                nbsc += ","
        dic["nbsc"] = nbsc
        # update B and N matrices
        for i in B_set:
            B = np.concatenate((B, A_conca[i, :]))
        for j in N_set:
            N = np.concatenate((N, A_conca[j, :]))
        B = B[range(len(B_set), len(B_set)+len(B_set)), :]
        N = N[range(len(N_set), len(N_set)+len(N_set)), :]
        dic["bscm"] = texMatrix(B)
        dic["nbscm"] = texMatrix(N)
        iteration += 1
        with open("solution.tex", "a") as f:
            f.write(content % dic)


def primalSimplex(N_set, B_set, N, B, z_N_star, x_B_star, A_conca):

    c = 0 - z_N_star
    A = N
    b = x_B_star
    n = len(N_set)
    #
    #   print the original dictionary
    #
    content = r'''\section{Input}\begin{alignat*}{3}max\quad%(obj)s\\s.t.\quad%(cstr)s\end{alignat*}'''

    x = []
    for i in range(1, c.shape[1] + 1):
        x.append("x_"+str(i))

    dic = {}

    obj = ""
    i = 0
    for val in np.nditer(c):
        obj += "%.2f%s" % (val, x[i])
        if (i + 1 < c.shape[1]):
            i += 1
            obj += "&+"
    dic["obj"] = obj

    cstr = ""
    A_list = A.tolist()
    b_list = b.tolist()
    l_j = len(A_list)
    l_i = len(A_list[0])
    i, j = 0, 0
    while 1:
        cstr += "%.2f%s" % (A_list[j][i], x[j])
        if (j + 1 < l_j):
            j += 1
            cstr += "&+"
        else:
            cstr += "&&\\leq%.2f" % b_list[0][i]
            i += 1
            if (i >= l_i):
                break
            cstr += "\\\\"
            j = (j + 1) % l_j
    cstr += ""
    dic["cstr"] = cstr
    with open("solution.tex", "a") as f:
        f.write(content % dic)
    #
    #   write the primal simplex
    #
    dic = {}

    with open("solution.tex", "a") as f:
        f.write("\n\\section{Calculation}\nThe matrix A is given by")

    content = r'''\[\begin{bmatrix}%(matrix)s\end{bmatrix}\]'''

    dic["matrix"] = texMatrix(A_conca)
    with open("solution.tex", "a") as f:
        f.write(content % dic)

    content = r'''
\\The initial sets of basic and non-basic indices are
\[\mathcal{B}=\{%(bsc)s\}\]
\[\mathcal{N}=\{%(nbsc)s\}\]
Corresponding to these sets, we have the submatrices of A:
\[B=\begin{bmatrix}%(bscm)s\end{bmatrix}\]
\[N=\begin{bmatrix}%(nbscm)s\end{bmatrix}\]
'''

    bsc = ""
    for i in range(len(B_set)):
        bsc += str(B_set[i])
        if i != len(B_set) - 1:
            bsc += ","
    dic["bsc"] = bsc

    nbsc = ""
    for i in range(len(N_set)):
        nbsc += str(N_set[i])
        if i != len(N_set) - 1:
            nbsc += ","
    dic["nbsc"] = nbsc

    dic["bscm"] = texMatrix(B)
    dic["nbscm"] = texMatrix(N)

    with open("solution.tex", "a") as f:
        f.write(content % dic)

    iteration = 0
    # now we enter the loop
    while True:
        # find the min in zN*
        m = 0
        for val in z_N_star.tolist()[0]:
            if val < m:
                m = val
        # if no negative in zN*, then optimal
        if m >= 0:
            # store the solution into a dict, key being the index, data being
            # the value
            content = r'''\subsection{Optimal solution:}\[%(solu)s\]\[%(optobj)s\]'''
            sol = {}
            solu = ''

            for i in range(len(N_set)):
                sol[N_set[i]] = z_N_star.tolist()[0][i]
            for j in range(len(B_set)):
                sol[B_set[j]] = x_B_star.tolist()[0][j]
            # print out solution, N.B. first then B.
            for key in range(A_conca.shape[0]):
                if key < n:
                    solu += "x_%i = %.2f " % (key + 1, sol[key])
                else:
                    solu += "w_%i = %.2f " % (key + 1 - n, sol[key])
                if key != A_conca.shape[0] - 1:
                    solu += "\\quad "
            dic["solu"] = solu
            # compute c_B
            c_B = []
            for i in B_set:
                if i < n:
                    c_B.append(c[0, i])
                else:
                    c_B.append(0)
            # compute the objective
            z = np.dot(x_B_star, np.transpose(c_B)).tolist()[0][0]
            optobj = ''
            for i in range(len(c_B)):
                optobj += "%.2f * %.2f " % (c_B[i], sol[B_set[i]])
                if i != len(c_B) - 1:
                    optobj += '+ '
            optobj += "= %.2f\n" % z
            dic["optobj"] = optobj
            with open("solution.tex", "a") as f:
                f.write(content % dic)
            return z

        content = r'''
\subsection{Iteration %(iter)s:}
\[j = %(j)s\]
\[e_{j} = \begin{bmatrix}%(e_j)s\end{bmatrix}\]
\[\Delta X_{\mathcal B} = B^{-1} N e_j = \begin{bmatrix}%(Binv)s\end{bmatrix} \begin{bmatrix}%(N)s\end{bmatrix} \begin{bmatrix}%(e_j)s\end{bmatrix} = \begin{bmatrix}%(del_x_B)s\end{bmatrix}\]
\[t = (max\{%(div)s\})^{-1}= %(t)s\]
\[i = %(i)s\]
\[e_{i} = \begin{bmatrix}%(e_i)s\end{bmatrix}\]
\[\Delta Z_{\mathcal N} = -(B^{-1} N)^{T} e_i = -(\begin{bmatrix}%(Binv)s\end{bmatrix} \begin{bmatrix}%(N)s\end{bmatrix})^{T} \begin{bmatrix}%(e_i)s\end{bmatrix} = \begin{bmatrix}%(del_z_N)s\end{bmatrix}\]
\[s = \frac{Z_{j}^{*}}{\Delta Z_{j}}= %(s)s\]
\[X_{\mathcal B}^{*} = X_{\mathcal B}^{*} - t \Delta X_{\mathcal B} + t * e_i = \begin{bmatrix}%(xB*)s\end{bmatrix}\]
\[Z_{\mathcal N}^{*} = Z_{\mathcal N}^{*} - s \Delta Z_{\mathcal N} + s * e_j = \begin{bmatrix}%(zN*)s\end{bmatrix}\]
\[\mathcal{B}=\{%(bsc)s\}\]
\[\mathcal{N}=\{%(nbsc)s\}\]
\[B=\begin{bmatrix}%(bscm)s\end{bmatrix}N=\begin{bmatrix}%(nbscm)s\end{bmatrix}\]
'''
        dic["iter"] = iteration + 1
        dic["Binv"] = texMatrix(np.linalg.inv(B))
        dic["N"] = texMatrix(N)
        # entering index
        j = z_N_star.tolist()[0].index(m)
        dic["j"] = j
        # construct e_j
        e_j = np.zeros(len(N_set))
        e_j[j] = 1
        e_j = np.matrix(e_j)
        dic["e_j"] = texMatrix(e_j)
        # compute del_x_B
        del_x_B = np.dot(e_j, np.dot(N, np.linalg.inv(B)))
        dic["del_x_B"] = texMatrix(del_x_B)
        # compute t and leaving index i
        v_t = []
        div = ''
        for i in range(x_B_star.shape[1]):
            if x_B_star[0, i] == 0:
                # convention for 0/0
                if del_x_B[0, i] == 0:
                    v_t.append(0)
                # +/- inf
                else:
                    v_t.append(del_x_B[0, i] > 0 and np.inf or 0 - np.inf)
            else:
                v_t.append(del_x_B[0, i] / x_B_star[0, i])
            div += '\\frac{%.1f}{%.1f}' % (del_x_B[0, i], x_B_star[0, i])
            if i != x_B_star.shape[1] - 1:
                div += ','
        dic["div"] = div
        t = max(v_t)
        if t <= 0:
            return "unbounded"
        i = v_t.index(t)
        dic["i"] = i
        t = 1 / t
        dic["t"] = t
        # construct e_i
        e_i = np.zeros(len(B_set))
        e_i[i] = 1
        e_i = np.matrix(e_i)
        dic["e_i"] = texMatrix(e_i)
        # compute del_z_N
        del_z_N = 0 - np.dot(e_i, np.transpose(np.dot(N, np.linalg.inv(B))))
        dic["del_z_N"] = texMatrix(del_z_N)
        # compute s
        s = z_N_star[0, j] / del_z_N[0, j]
        dic["s"] = s
        # update xB* and zN*
        x_B_star = x_B_star - t * del_x_B
        x_B_star[0, i] = t
        dic["xB*"] = texMatrix(x_B_star)
        z_N_star = z_N_star - s * del_z_N
        z_N_star[0, j] = s
        dic["zN*"] = texMatrix(z_N_star)
        # update B and N set
        tmp = B_set[i]
        B_set[i] = N_set[j]
        N_set[j] = tmp

        bsc = ""
        for i in range(len(B_set)):
            bsc += str(B_set[i])
            if i != len(B_set) - 1:
                bsc += ","
        dic["bsc"] = bsc

        nbsc = ""
        for i in range(len(N_set)):
            nbsc += str(N_set[i])
            if i != len(N_set) - 1:
                nbsc += ","
        dic["nbsc"] = nbsc
        # update B and N matrices
        for j in B_set:
            B = np.concatenate((B, A_conca[j, :]))
        for i in N_set:
            N = np.concatenate((N, A_conca[i, :]))
        B = B[range(len(B_set), len(B_set)+len(B_set)), :]
        N = N[range(len(N_set), len(N_set)+len(N_set)), :]
        dic["bscm"] = texMatrix(B)
        dic["nbscm"] = texMatrix(N)
        iteration += 1
        with open("solution.tex", "a") as f:
            f.write(content % dic)


def readInput():
    # global A, b, c
    if len(sys.argv) < 3:
        print("Usage: $ python3 Simplex.py my_A.csv my_c.csv my_b.csv")
        exit(-1)
    # "c" vector
    data = np.loadtxt(sys.argv[2], delimiter=",")
    c = np.matrix(data)

    # "b" vector
    data = np.loadtxt(sys.argv[3], delimiter=",")
    b = np.matrix(data)

    # "A" matrix
    A_col = c.shape[1]
    A_row = b.shape[1]
    data = np.loadtxt(sys.argv[1], delimiter=",")
    A = np.matrix(data)
    A = A.reshape((A_col, -1))
    # check A dimention
    if (A.shape[1] != A_row):
        print("ERROR: A, b & c do not match in dimentions!")
        exit(-1)

    return [A, b, c]


def texMatrix(m):
    s = ''
    for i in range(m.shape[1]):
        for j in range(m.shape[0]):
            s += str(m[j, i])
            if j != m.shape[0] - 1:
                s += "&"
        s += '\\\\'
    return s

if __name__ == '__main__':
    main()
