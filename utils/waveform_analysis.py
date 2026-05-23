#!/usr/bin/env python3

__author__ = "Bong-Hwi Lim, Luca Aglietta, Roberto Russo"
__maintainer__ = "Bong-Hwi Lim, Luca Aglietta, Roberto Russo"
__email__ = "bong-hwi.lim@cern.ch, luca.aglietta@edu.unito.it, r.russo@cern.ch"
__status__ = "Development"

# in the following set of functions, a graph object is considered as a list of two arrays of the same length (typically time for the x array and voltage for the y array) 

import numpy as np


def GetBaseline(inputGraph, nbins=50, startBin=0):
    if len(inputGraph[0]) != len(inputGraph[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    if startBin + nbins > len(inputGraph[1]):
        startBin = 0
    baseline_evaluation_array = inputGraph[1][startBin:startBin+nbins]
    baseline = np.mean(baseline_evaluation_array)
    baseline_rms = np.std(baseline_evaluation_array)
    return baseline, baseline_rms


def GetMinPointIndexY(inputGraph):
    return np.argmin(inputGraph[1])


def GetMinY(inputGraph):
    return np.min(inputGraph[1])


def hasSignal(inputGraph, threshold=10):
    if len(inputGraph[0]) != len(inputGraph[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    baseline, _ = GetBaseline(inputGraph)
    minvalue = GetMinY(inputGraph)
 #   print(f"Baseline: {baseline}, Minimum value: {GetMinY(inputGraph)}, Threshold: {threshold}")
    return abs(baseline - minvalue) > threshold


def FindLeftNearBaseline(inputGraph, cut, pointsWithinCut=10, totalStep=10):
    leftIndex = -999
    baseline, _ = GetBaseline(inputGraph)
    minIndex = GetMinPointIndexY(inputGraph)
    countWithinCut = 0
    for i in range(minIndex):
        if abs(inputGraph[1][i] - baseline) > cut:
            countWithinCut += 1
        else:
            countWithinCut = 0
        if countWithinCut > pointsWithinCut:
            leftIndex = i - totalStep
            break
    return leftIndex


def derivative_rec(data, npoints=500):
    nframes = len(data)
    der = np.zeros((nframes))
    sp = 0
    sm = 0
    for i in range(nframes):
        der[i] = ((sp-sm)/npoints)
        if i<=npoints and (2*i)<=(nframes-1):
            sp = sp + data[2*i+-1] + data[2*i] - data[i]
            sm = sm + data[i-1] 
        elif i>npoints and (i+npoints)<=(nframes-1):
            sp = sp + data[i+npoints] - data[i]
            sm = sm + data[i-1] - data[i-npoints-1]
        elif (i+npoints)>(nframes-1) and (2*i)>(nframes-1):
            sp = sp - data[i]
            sm = sm + data[i-1] - data[2*i-nframes] - data[2*i-nframes-1] 
    return der


def find_edge(inputGraph, dt_ns, t_int=17.5, thr=1):
    if len(inputGraph[0]) != len(inputGraph[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    npoints = round(t_int/dt_ns) 
    der = derivative_rec(inputGraph[1], npoints)
    val = np.min(der)
    if val < -thr:
        leftIndex = np.argmin(der)
    else:
        leftIndex = -999 
    return leftIndex


def PointDer(data, i, npoints):
    sp = np.sum(data[i:i+npoints])
    sm = np.sum(data[i-npoints:i])
    return sp,sm,-1*(sp - sm)


def DerAmp(inputGraph, t0_bin, dt_ns, int1, int2):
    if len(inputGraph[0]) != len(inputGraph[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    n = round(max(int1, int2)/dt_ns)
    p = round(min(int1, int2)/dt_ns)
    spn, smn, dn = PointDer(inputGraph[1],t0_bin,n)
    spp, smp, dp = PointDer(inputGraph[1],t0_bin,p)
    baseline = (smn - smp)/(n-p)
    underline = (spn - spp)/(n-p)
    amplitude = (dn - dp)/(n-p)
    return baseline, underline, amplitude


def GetAmp(inputGraph , t0_bin, dt_ns, int1, int2):
    if len(inputGraph[0]) != len(inputGraph[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    n = round(max(int1, int2)/dt_ns)
    p = round(min(int1, int2)/dt_ns)
    baseline = np.mean(inputGraph[1][t0_bin-n:t0_bin-p])
    baseline_rms = np.std(inputGraph[1][t0_bin-n:t0_bin-p])
    baseline_noise_point = inputGraph[1][t0_bin-p]  # noise is evaluated by looking at the closest point to the signal among those used to evaluate the baseline
    underline = np.mean(inputGraph[1][t0_bin+p:t0_bin+n])
    underline_rms = np.std(inputGraph[1][t0_bin+p:t0_bin+n])
    underline_noise_point = inputGraph[1][t0_bin+p]  # noise is evaluated by looking at the closest point to the signal among those used to evaluate the underline
    amplitude = baseline - underline
    return baseline, baseline_rms, baseline_noise_point, underline, underline_rms, underline_noise_point, amplitude


def pol1(x, m, q):
    f = q + m * x
    return f


def linearFit(inputGraph, fitRange):
    from lmfit import Model
    model = Model(pol1)
    model.set_param_hint('m', value=-5)    # to be improved
    model.set_param_hint('q', value=1000)  # to be improved
    pars = model.make_params()
    idx_fit = np.where((inputGraph[0] > fitRange[0]) & (inputGraph[0] < fitRange[1]))[0]
    xfit = inputGraph[0][idx_fit]
    yfit = inputGraph[1][idx_fit]
    result = model.fit(yfit, x=xfit, params=pars)
    return result


def invSigmoid(x, t0, c, delta, alpha):
    f = c - delta/(1 + np.exp(-1 * alpha * (x - t0)))
    return f


def sigmoidFit(inputGraph, fitRange, t0=-1, baseLine=-1, underLine=-1):
    from lmfit import Model
    model = Model(invSigmoid)
    if baseLine == -1 | underLine == -1:
        baseLine, _ = GetBaseline(inputGraph, 400, fitRange[0])
        underLine = baseLine - GetMinY(inputGraph)
    if t0 == -1:
        t0 = inputGraph[0][FindLeftNearBaseline(inputGraph, cut=3.)]
    model.set_param_hint('t0', value=t0)
    model.set_param_hint('c', value=baseLine, min=baseLine*0.9, max=baseLine*1.1)
    model.set_param_hint('delta', value=(baseLine-underLine), min=(baseLine-underLine)*0.8, max=(baseLine-underLine)*1.2)
    model.set_param_hint('alpha', value=1.)  # to be optimized
    pars = model.make_params()
    idx_fit = np.where((inputGraph[0] > fitRange[0]) & (inputGraph[0] < fitRange[1]))[0]
    xfit = inputGraph[0][idx_fit]
    yfit = inputGraph[1][idx_fit]
    result = model.fit(yfit, x=xfit, params=pars)
    return result


def incompleteGammaFunction(x, t0, c, delta, alpha, beta):
    from scipy.special import gammainc
    f = np.piecewise(x, [x < t0, x >= t0], [c, lambda x: c - delta * gammainc(alpha, (x - t0) / beta)])
    return f


def incompGammaFit(inputGraph, fitRange, t0=-1, baseLine=-1, underLine=-1):
    from lmfit import Model
    model = Model(incompleteGammaFunction)
    if baseLine == -1 or underLine == -1:
        baseLine, _ = GetBaseline(inputGraph, 400, fitRange[0])
        underLine = baseLine - GetMinY(inputGraph)
    if t0 == -1:
        t0 = inputGraph[0][FindLeftNearBaseline(inputGraph, cut=5.)]
    model.set_param_hint('t0', value=t0)
    model.set_param_hint('c', value=baseLine, min=baseLine*0.9, max=baseLine*1.1)
    model.set_param_hint('delta', value=(baseLine-underLine), min=(baseLine-underLine)*0.8, max=(baseLine-underLine)*1.2)
    model.set_param_hint('alpha', value=0.5, min=0., max=3.)
    model.set_param_hint('beta', value=0.5, min=0., max=10.)
    pars = model.make_params()
    idx_fit = np.where((inputGraph[0] > fitRange[0]) & (inputGraph[0] < fitRange[1]))[0]
    xfit = inputGraph[0][idx_fit]
    yfit = inputGraph[1][idx_fit]
    result = model.fit(yfit, x=xfit, params=pars)
    return result


def GetCustomRatioPointIndex(fitFunctionName: str, fitResult, scaleFactorX, ratio=0.5, currentBaseline=-1, minValue=-1, start=-1, end=-1):
    from scipy.optimize import brentq
    if "Sigmoid" in fitFunctionName:
        if len(fitResult.params.keys()) != 4:
            raise ValueError("Needed 4 parameters for sigmoid function.")
        middle = fitResult.params['t0'].value
        baseline_value = fitResult.params['c'].value
        ampl = fitResult.params['delta'].value
        alph = fitResult.params['alpha'].value
        underline_value = invSigmoid(1E9, middle, baseline_value, ampl, alph)  # some big value
        ratio_value = (baseline_value - underline_value) * ratio
        starting = middle - 100 * scaleFactorX
        ending = middle + 480 * scaleFactorX
        if start != -1:
            starting = start
        if end != -1:
            ending = end
        index_half = brentq(f=invSigmoid, a=starting, b=ending, xtol=1E-6, args=(middle, ratio_value, ampl, alph))
        values = [index_half, ratio_value]
        return values
    elif "incompGamma" in fitFunctionName:
        if len(fitResult.params.keys()) != 5:
            raise ValueError("Needed 5 parameters for incomplete gamma function.")
        first = fitResult.params['t0'].value
        delt = fitResult.params['delta'].value
        alph = fitResult.params['alpha'].value
        bet = fitResult.params['beta'].value
        if currentBaseline == -1:
            currentBaseline = fitResult.params['c'].value
        if minValue == -1:
            minValue = incompleteGammaFunction(1E9, first, currentBaseline, delt, alph, bet)
        ampl = currentBaseline - minValue
        ratio_value = ampl * ratio
        starting = first - 100 * scaleFactorX
        ending = first + 480 * scaleFactorX
        if start != -1:
            starting = start
        if end != -1:
            ending = end
        index_half = brentq(f=incompleteGammaFunction, a=starting, b=ending, xtol=1E-6, args=(first, ratio_value, ampl, alph, bet))
        values = [index_half, ratio_value]
        return values
    elif "linearFit" in fitFunctionName:
        if len(fitResult.params.keys()) != 2:
            raise ValueError("Needed 2 parameters for linear function.")
        if currentBaseline == -1 or minValue == -1:
            raise ValueError("currentBaseline and minValue must be provided for linearFit.")
        starting = -5
        ending = 15
        if start != -1:
            starting = start
        if end != -1:
            ending = end
        q = fitResult.params['q'].value
        m = fitResult.params['m'].value
        searchValue = abs((currentBaseline - minValue)) * ratio
        index_half = brentq(f=pol1, a=starting, b=ending, xtol=1E-6, args=(searchValue, m))
        values = [index_half, ratio]
        return values
    else:
        print("Only Sigmoid, incompGamma, linearFit are implemented. The chosen function is none of them. Will return empty result.")
        return []


def getCFDpoint(inputGraph, ratioValue=0.5, findRange=[1, 2], currentBaseline=-1, minValue=-1):
    if len(inputGraph[0]) != len(inputGraph[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    findBelow = True
    findAbove = False
    values=[]
    returnValues=[]
    if currentBaseline == -1:
        currentBaseline, _ = GetBaseline(inputGraph, 400, findRange[0])
    if minValue == -1:
        minValue = currentBaseline - GetMinY(inputGraph)
    searchValue = currentBaseline - abs(currentBaseline - minValue) * ratioValue
    for i in np.arange(start=findRange[0], stop=findRange[1], step=1):
        if findBelow:
            if inputGraph[1][i] < searchValue:
                values.append(inputGraph[0][i])
                findBelow = False
                findAbove = True
        if findAbove:
            if inputGraph[1][i] > searchValue:
                values.append(inputGraph[0][i])
                findBelow = True
                findAbove = False
    if len(values) % 2 == 0:
        return [-999, -999]
    if len(values) == 1:
        returnValues.append(values[0])
        returnValues.append(0)
        return returnValues
    if len(values) > 2:
        distance = values[-1] - values[0]
        middle = values[0] + distance/2
        returnValues.append(middle)
        returnValues.append(distance)
        return returnValues
    return [-999, -999]


def getCFDpoint2(inputGraph, ratioValue=0.5, findRange=[1, 2], currentBaseline=-1, minValue=-1):
    if len(inputGraph[0]) != len(inputGraph[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    # Used average
    findBelow=True
    findAbove=False
    values=[]
    returnValues=[]
    if currentBaseline == -1:
        currentBaseline, _ = GetBaseline(inputGraph, 400, findRange[0])
    if minValue == -1:
        minValue = currentBaseline - GetMinY(inputGraph)
    searchValue = currentBaseline - abs((currentBaseline - minValue)) * ratioValue
    for i in np.arange(start=findRange[0], stop=findRange[1], step=1):
        if findBelow:
            if inputGraph[1][i] < searchValue:
                values.append((inputGraph[0][i-1] + inputGraph[0][i])/2)  # average of the two points
                findBelow = False
                findAbove = True
        if findAbove:
            if inputGraph[1][i] > searchValue:
                values.append((inputGraph[0][i-1] + inputGraph[0][i])/2)  # average of the two points
                findBelow = True
                findAbove = False
    if len(values) % 2 == 0:
        return [-999, -999]
    if len(values) == 1:
        returnValues.append(values[0])
        returnValues.append(0)
        return returnValues
    if len(values) > 2:
        distance = values[-1] - values[0]
        middle = values[0] + distance/2
        returnValues.append(middle)
        returnValues.append(distance)
        return returnValues
    return [-999, -999]


def RunningAverage(graphG, nAvg=2):
    if len(graphG[0]) != len(graphG[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    g_runavg_x = np.zeros(len(graphG[0])-nAvg)
    g_runavg_y = np.zeros(len(graphG[1])-nAvg)
    for iBin in range(len(g_runavg_x)):
        left_index = iBin-nAvg+1 if iBin+1>nAvg else 0
        right_index = iBin+2 if iBin+2<len(graphG[0]) else len(graphG[0])+1
        nPoints = right_index - left_index
        avgx = np.sum(graphG[0][left_index:right_index])/nPoints
        avgy = np.sum(graphG[1][left_index:right_index])/nPoints
        # Set the point of the running average graph
        if (iBin >= nAvg-1):
            g_runavg_x[iBin] = avgx
            g_runavg_y[iBin] = avgy 
        else:
            g_runavg_x[iBin] = graphG[0][iBin]
            g_runavg_y[iBin] = graphG[1][iBin]
    return [g_runavg_x, g_runavg_y]


def Smooth(graphG, nsm=2):
    if len(graphG[0]) != len(graphG[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    xsm = graphG[0][nsm:graphG[0].size-nsm]
    ysm = np.zeros(graphG[0].size-2*nsm)
    for elem in range(nsm,graphG[0].size-nsm):
        ysm[elem-nsm] = (np.sum(graphG[1][elem-nsm:elem+nsm+1]))/(2*nsm+1)
    return [xsm, ysm]


def ResampleGraph(graphG, findRange=[-1,-1], softwareScaleFactorX=0.0125):
    from scipy.interpolate import CubicSpline
    if len(graphG[0]) != len(graphG[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    if len(findRange) > 2:
        raise ValueError("findRange must contain two numbers only: the first and last point indexes where you want to resample your waveform.")
    if ((findRange[0] == -1) & (findRange[1] == -1)):
        leftEdge = FindLeftNearBaseline(graphG, cut=7, pointsWithinCut=10, totalStep=10)
        findRange[0] = leftEdge
        findRange[1] = leftEdge + 100
    x = graphG[0][findRange[0]:findRange[1]+1]
    y = graphG[1][findRange[0]:findRange[1]+1]
    cs = CubicSpline(x, y)
    xs = np.arange(start=graphG[0][findRange[0]], stop=graphG[0][findRange[1]]+softwareScaleFactorX, step=softwareScaleFactorX)
    return [xs, cs(xs)]


def GetMaxX(graphG):
    return np.max(graphG[0])


def ComputeDerivative(graphG, j, npts=5):
    if len(graphG[0]) != len(graphG[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    if npts < 3:
        raise ValueError("You want to compute the derivative over less than 3 points. They are too few.")
    if npts == 3:
        der = (graphG[1][j+1] - graphG[1][j-1])/(graphG[0][j+1] - graphG[0][j-1])
    else:
        der = (graphG[1][j-2] - 8 * graphG[1][j-1] + 8 * graphG[1][j+1] - graphG[1][j+2])/(graphG[0][j+2] - graphG[0][j+1])/12.
    return der


def GetDerivative(graphG, nsm=4):
    xgder = []
    ygder = []
    for j, x in enumerate(graphG[0][2:(-1-nsm-2)]):
        der = 0
        nnn = 0
        k = 0
        while k <= nsm:
            der += ComputeDerivative(graphG, j+k)
            nnn += 1.
            k += 1
        if (nnn > 0):
            der /= nnn
            xgder.append(x)
            ygder.append(der)
    return [np.asarray(xgder), np.asarray(ygder)]


def CountNextNegativeDer(graphG):
    xarrayn = []
    yarrayn = []
    for j in range(len(graphG[1])-2):
        x = graphG[0][j]
        cntneg = 0
        k = j
        while k < len(graphG[1])-2:
            der = ComputeDerivative(graphG, k)
            if (der >= 0):
                break
            cntneg+=1
            k += 1
        xarrayn.append(x)
        yarrayn.append(cntneg)
    return [np.asarray(xarrayn), np.asarray(yarrayn)]


# used by function FindEdge
def GetMeanAndRMSCounts(graphG, xmin, xmax):
    if len(graphG[0]) != len(graphG[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    summ = 0
    sum2 = 0
    cnts = 0
    for j in range(len(graphG[1])):
        x = graphG[0][j]
        c = graphG[1][j]
        if ((x > xmin) & (x < xmax)):
            cnts += 1.
            summ += c
            sum2 += (c * c)
    if cnts > 0:
        mean = summ / cnts
        rms = np.sqrt(sum2 / cnts - mean * mean)
    else:
        mean = 0
        rms = 0
    return mean, rms


def interpolate_x_value(graph, zero_crossing_index, interpolation_points):
    left_index = (zero_crossing_index-interpolation_points) if zero_crossing_index > interpolation_points else 0
    right_index = (zero_crossing_index+interpolation_points+1) if (zero_crossing_index+interpolation_points) <= len(graph[0]) else (len(graph[0])+1)
    xG = graph[0][left_index:right_index]
    yG = graph[1][left_index:right_index]
    sumx = np.sum(xG)
    sumy = np.sum(yG)
    sumxy = np.sum(xG * yG)
    sumx2 = np.sum(xG**2)
    npts = right_index - left_index
    m = (npts * sumxy - sumx * sumy) / (npts * sumx2 - sumx * sumx)
    q = (sumy * sumx2 - sumx * sumxy) / (npts * sumx2 - sumx * sumx)
    return m, q


def FindOnGraph(graphGcount, y, xmin, xmax, interpolate, dx, backw=False):
    if len(graphGcount[0]) != len(graphGcount[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    idx_min = (np.abs(graphGcount[0] - xmin)).argmin()-1
    idx_min = idx_min if idx_min>0 else 0
    idx_max = (np.abs(graphGcount[0] - xmax)).argmin()+1
    idx_max = idx_max if idx_max < len(graphGcount[0]) else len(graphGcount[0])-1
    lookup_index = 0
    if backw:
        lookup_index = -1
    zero_crossings = (np.where(np.diff(np.signbit(graphGcount[1][idx_min:idx_max+1]-y)))[0])+1  # find indexes where the signal is crossed by the value y
    if len(zero_crossings) > 0:
        index = idx_min+zero_crossings[lookup_index]
        # look within a 10 sampled points interval within zero crossing to find the point satisfying the condition at line 559
        jfirst = index-5 if index>5 else 0
        dstep = 1
        if backw:
            jfirst = index+5 if index+5<len(graphGcount[0]) else len(graphGcount[0])
            dstep = -1
        for jstep in range(10):
            j = jfirst + dstep * jstep
            if ((j>=0) and (j<len(graphGcount[0]))):
                index_point = j
            elif j<0:
                index_point = 0
            else:
                index_point = len(graphGcount[0])-1
            c = graphGcount[1][index_point]
            x = graphGcount[0][index_point]
            # point before
            if (((j-dstep)>=0) and ((j-dstep)<len(graphGcount[0]))):
                index_before = j-dstep
            elif ((j-dstep)<0):
                index_before = 0
            else:
                index_before = len(graphGcount[1])-1
            cbef = graphGcount[1][index_before]
            # point after
            if (((j+dstep)>=0) and ((j+dstep)<len(graphGcount[0]))):
                index_after = j+dstep
            elif ((j+dstep)<0):
                index_after = 0
            else:
                index_after = len(graphGcount[1])-1
            caft = graphGcount[1][index_after]
            if ((dstep == 1 and c < y and cbef > y and caft < y) or (dstep == -1 and c > y and cbef < y and caft > y)):
                if interpolate == 0:
                    return x
                m, q = interpolate_x_value(graphGcount, j, interpolate)
                with np.errstate(divide='ignore', invalid='ignore'):  # suppress the warning message for 0 division
                    xinterp = (y - q)/m
                # last 2 or conditions are just to check the negative "monotonicity" of the signals
                if ((xinterp<xmin) or (xinterp>xmax) or (abs(xinterp-x)>dx) or ((c>y) and (x>xinterp)) or ((c<y) and (x<xinterp))):
                    continue
                return xinterp
        # if none of the points satisfies the condition, use the rough zero crossing 
        x = graphGcount[0][index]
        if interpolate == 0:
            return x
        if graphGcount[1][index] < y:
            x1 = graphGcount[0][index-1]
            y1 = graphGcount[1][index-1]
            x2 = graphGcount[0][index]
            y2 = graphGcount[1][index]
        else:
            x1 = graphGcount[0][index]
            y1 = graphGcount[1][index]
            x2 = graphGcount[0][index+1]
            y2 = graphGcount[1][index+1]
        m = (y2-y1)/(x2-x1)
        q = y2 - m*x2
        x = (y - q)/m
        return x
    return -999.


def FindEdge(graphGcount, graphGnegDer, graphGder):
    ## First very rough step: compute flat levels on the left and on the right and check their difference
    maxTime = GetMaxX(graphGcount)
    levleft, rmsleft = GetMeanAndRMSCounts(graphGcount, 0., 2000.)
    levright, rmsright = GetMeanAndRMSCounts(graphGcount, maxTime - 2000, maxTime)
    y50 = 0.5 * (levleft + levright)
    t50fromleft = FindOnGraph(graphGcount, y50, 0., maxTime, 4)
    t50fromright = FindOnGraph(graphGcount, y50, 0., maxTime, 4, True)
    roughsig = levleft - levright
    # print(f"Rough signal = {roughsig} Rough edge position = {t50fromleft} {t50fromright}\n")
    minSearchWindow = 0
    maxSearchWindow = maxTime
    if roughsig > 0.0005:
        minSearchWindow = min(t50fromleft, t50fromright) - 6000.
        minSearchWindow = max(minSearchWindow, 0)
        maxSearchWindow = max(t50fromleft, t50fromright) + 6000.
        maxSearchWindow = min(maxSearchWindow, maxTime)
    # print(f"Search window = {minSearchWindow} {maxSearchWindow}\n")
    ## Second step: search for accumulation of adjacent points with negative
    # derivative
    xmaxn = -1
    cmaxn = -1
    jmaxn = -1
    if len(graphGnegDer[0]) != len(graphGnegDer[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    if len(graphGnegDer[0]) > 0:
        for j in range(len(graphGnegDer[0])):
            x = graphGnegDer[0][j]
            c = graphGnegDer[1][j]
            if ((x < minSearchWindow) or (x > maxSearchWindow)):
                continue
            if c > cmaxn:
                cmaxn = c
                xmaxn = x
                jmaxn = j
            if c == cmaxn:
                sum0 = 0
                sum1 = 0
                for k in range(1,20):
                    sum0 += graphGnegDer[1][jmaxn+k]
                    sum1 += graphGnegDer[1][j+k]
                if sum1 > sum0:
                    cmaxn = c
                    xmaxn = x
                    jmaxn = j
        # print(f"Maximum adjacent points with negative derivative: t_maxn={xmaxn} n_neg={cmaxn}\n")
    ## Third step: search for minimum of derivative and range where derivative differs from 0
    # xminder = -1
    dermin = 99999.
    jminder = -1
    if len(graphGder[0]) != len(graphGder[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    for j in range(len(graphGder[0])):
        x = graphGder[0][j]
        d = graphGder[1][j]
        if ((x < minSearchWindow) or (x > maxSearchWindow)):
            continue
        if d < dermin:
            dermin = d
            # xminder = x
            jminder = j
    if jminder < 0:
        endplateau = 0
        edgeleft = 0
        edgeright = 0
        return endplateau, edgeleft, edgeright
    # print(f"Minimum of derivative: xminder={xminder} dermin={dermin}\n")
    jleft = -1
    dthresh = -1E-7
    j = jminder
    while j > 0:
        if graphGder[1][j] > dthresh:
            jleft = j
            break
        j -= 1
    jright = -1
    for j in range(jminder, len(graphGder[0])):
        if graphGder[1][j] > dthresh:
            jright = j
            break
    xleft = graphGder[0][jleft]
    xright = graphGder[0][jright]
    # print(f"Region of negative derivative: xleft={xleft} xright={xright}\n")
    if ((xmaxn > 0) and (abs(xmaxn - xleft) < 5000) and (abs(xmaxn - xright) < 5000)):
        xleft = min(xleft, xmaxn)
        xright = max(xright, xmaxn)
    # print(f"Edge range after analysis of derivative: xleft={xleft} xright={xright}\n")
    ## Fourth step: start from left and search for N points with couns < baseline- 3 * sigma
    cmean, crms = GetMeanAndRMSCounts(graphGcount, 0., xleft)
    # print(f"Mean before edge = {cmean} rms = {crms}\n")
    thresh = cmean - 3 * crms
    threshbin = np.rint(max(10., cmaxn / 3.))
    if cmaxn < 0:
        threshbin = 3
    xleft2 = -1.
    for j, x in enumerate(graphGcount[0]):
        nbelow = 0
        for c2 in graphGcount[1][j:-1]:
            if c2 < thresh:
                nbelow += 1
            else:
                break
        if nbelow > threshbin:
            xleft2 = x
            break
    # print(f"Left Edge from baseline-N*rms = {xleft2}\n")
    if xleft2 > 0:
        endplateau = min(xleft, xleft2)
        edgeleft = max(xleft, xleft2)
        edgeright = xright
        # print(f"Edge range after all steps: endplateau={endplateau} edgeleft={edgeleft} edgeright={edgeright}\n")
        return endplateau, edgeleft, edgeright
    endplateau = 0
    edgeleft = 0
    edgeright = 0
    return endplateau, edgeleft, edgeright


## Method finalisation: Following two functions are the approved methods to measure baseline and underline according the threshold-based t0 identification algorithm used for the analysis of SPS June '22 data
# function to get the baseline of the graph
def GetDefaultBaseline(inputGraph, t0_bin, dt_ns, evaluation_time_ns=2.5, start_time_before_t0_ns=17.5):  # evaluation_time_ns is the sampled time length to evaluate the baseline, start_time_before_t0_ns is the starting point before t0 to evaluate the baseline
    if len(inputGraph[0]) != len(inputGraph[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    evaluationPoints = int(np.rint(evaluation_time_ns/dt_ns))
    startPoint = int(np.rint(t0_bin-(start_time_before_t0_ns/dt_ns)))
    baseline, baseline_RMS = GetBaseline(inputGraph, nbins=evaluationPoints, startBin=startPoint)
    baseline_noise_point = inputGraph[1][startPoint+evaluationPoints]  # noise is evaluated by looking at the closest point to the signal among those used to evaluate the baseline
    return baseline, baseline_RMS, baseline_noise_point


# function to get the default underline of the graph
def GetDefaultUnderline(inputGraph, t0_bin, dt_ns, evaluation_time_ns=1.25, start_time_after_t0_ns=21.5):  # evaluation_time_ns is the sampled time length to evaluate the underline, start_time_after_t0_ns is the starting point after t0 to evaluate the baseline
    if len(inputGraph[0]) != len(inputGraph[1]):
        raise ValueError("Passed two arrays of different length for the same graph.")
    evaluationPoints = int(np.rint(evaluation_time_ns/dt_ns))
    startPoint = int(np.rint(t0_bin+(start_time_after_t0_ns/dt_ns)))
    underline, underline_RMS = GetBaseline(inputGraph, nbins=evaluationPoints, startBin=startPoint)
    underline_noise_point = inputGraph[1][startPoint]  # noise is evaluated by looking at the closest point to the signal among those used to evaluate the underline
    return underline, underline_RMS, underline_noise_point


# LGAD waveform analysis
# based on https://gitlab.cern.ch/alice-its3-wp3/opamp-utils/-/blob/master/QA/LGAD/analisiSEQ1.C
def lgad_signal_analysis(graph, SignalLeftRange=0.5, SignalRightRange=0.9, ChargeLeftRange=0.5, ChargeRightRange=0.9, BackgroundLeftRange=10):
    # Look for the maximum point in the array
    pos_rough = np.argmax(graph[1])
    amp_max_rough = np.max(graph[1])
    # define region where to look for signal
    SignalRegionMin = graph[0][pos_rough] - SignalLeftRange
    SignalRegionMax = graph[0][pos_rough] + SignalRightRange
    # compute noise mean and RMS in the region before actual signal and maximum and minimum values in the same region
    background_mask = np.where((graph[0] > (SignalRegionMin-BackgroundLeftRange)) & (graph[0] < SignalRegionMin))
    npointBkg = len(graph[0][background_mask])
    sumampBkg = np.sum(graph[1][background_mask])
    sumamp2Bkg = np.sum((graph[1][background_mask])**2)
    ampMaxBefore = np.max(graph[1][background_mask])
    ampMinBefore = np.min(graph[1][background_mask])
    bkg = sumampBkg/npointBkg
    rmsBkg = np.sqrt(sumamp2Bkg/npointBkg - bkg**2)
    # compute maximum amplitude
    maximum_signal_mask = np.where((graph[0] > SignalRegionMin) & (graph[0] < SignalRegionMax))
    ampMax = np.max(graph[1][maximum_signal_mask])
    time_ampMax = graph[0][maximum_signal_mask][np.argmax(graph[1][maximum_signal_mask])]
    pos_ampMax = np.where(graph[0] == time_ampMax)[0][0]
    # compute RMS background and subtract from baseline
    background_no_baseline_mask = np.where((graph[0] > (SignalRegionMin-BackgroundLeftRange)) & (graph[0] < SignalRegionMin))
    npointBkg = len(graph[0][background_no_baseline_mask]) 
    sumampBkg = np.sum(graph[1][background_no_baseline_mask]-bkg)
    sumamp2Bkg = np.sum((graph[1][background_no_baseline_mask]-bkg)**2)
    rmsBkg_noBaseline = np.sqrt(sumamp2Bkg/npointBkg - (sumampBkg/npointBkg)**2)
    ampMax_noBaseline = ampMax-bkg
    # SNR calculation
    SNR = ampMax_noBaseline/rmsBkg_noBaseline
    # compute the charge 
    ChargeRegionMin = graph[0][pos_rough] - ChargeLeftRange
    ChargeRegionMax = graph[0][pos_rough] + ChargeRightRange
    charge_mask = np.where((graph[0] > ChargeRegionMin) & (graph[0] < ChargeRegionMax))
    Charge = np.sum(graph[1][charge_mask])
    Charge_noBaseline = Charge-bkg
    # loop on CFD fractions
    CFDFractions = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    ampTh = np.zeros(len(CFDFractions))
    timeTh = np.zeros(len(CFDFractions))
    derTh = np.zeros(len(CFDFractions))
    jitterTh = np.zeros(len(CFDFractions))
    for b, CFD in enumerate(CFDFractions):
        pos_att = pos_ampMax
        while graph[0][pos_att] > SignalRegionMin:
            if (graph[1][pos_att]-bkg) < (ampMax_noBaseline*(CFD/100.)):
                ampTh[b] = graph[1][pos_att]-bkg
                timeTh[b] = graph[0][pos_att] + (graph[0][pos_att+1]-graph[0][pos_att])/(graph[1][pos_att+1]-graph[1][pos_att])*(ampMax_noBaseline*CFD/100.-(graph[1][pos_att]-bkg))
                derTh[b] = abs((graph[1][pos_att]-graph[1][pos_att+1])/(graph[0][pos_att]-graph[0][pos_att+1]))
                jitterTh[b] = rmsBkg_noBaseline/derTh[b]
                break
            pos_att -= 1
    return (ampMax, ampMax_noBaseline, bkg, rmsBkg, Charge, Charge_noBaseline, ampTh, timeTh, derTh, jitterTh)
