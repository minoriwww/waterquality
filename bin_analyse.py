import numpy as np, scipy.stats as st
from math import sqrt
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
# (number of correct predict, number of all predice, confidence level)
sns.set()

# https://www.jiqizhixin.com/articles/2018-07-06-5
z = 1.96 #for 95% confidence level
# st.norm.interval(0.95, loc=np.mean(s), scale=st.sem(s))
interval = z * sqrt( (0.1 * (1 - 0.1)) / 140)

from numpy.random import seed
seed(1)
from numpy.random import rand

from numpy.random import randint

from numpy import mean

from numpy import median

from numpy import percentile
from scipy.optimize import curve_fit
# seed the random number generator



err_bin10 = 10*[0.0,-0.5023155212402344,0.1515655517578125,0.7602171897888184,1.145859718322754,-3.516132354736328,-0.6084146499633789,-5.193119049072266,0.591475784778595,-1.881805419921875,1.8736810684204102,-0.2771425247192383,-0.8787617683410645,-0.17932367324829102,-0.6284408569335938,1.9549331665039062,0.28646063804626465,0.8224778175354004,-0.8758726119995117,0.19048357009887695,-1.626321792602539,-0.6578423976898193,0.9473075866699219,-1.8103580474853516,-1.5677146911621094,-3.1333436965942383,0.4643988609313965,2.502443313598633,-0.6599254608154297,-0.26084375381469727,-1.1500110626220703,0.9680152535438538,0.3467937707901001,-3.9340362548828125,-0.20847606658935547,0.1109389066696167,-1.7417497634887695,0.5872764587402344,-1.35711669921875,0.10165854543447495,1.234926700592041,-0.6065936088562012,-3.034621238708496,-2.0492992401123047,-0.4739856719970703,0.9423503875732422,2.274594306945801,-0.926966667175293,2.361504554748535,0.23089618980884552,0.18584060668945312,0.3538541793823242,0.04775381088256836,-1.9129486083984375,-1.6954717636108398,1.2165889739990234,0.15504932403564453,1.268141746520996,-2.262491226196289,-1.335136890411377,-0.5093350410461426,0.31419849395751953,0.7923250198364258,-0.02504730224609375,-0.8025712966918945,1.8563776016235352,0.4775829315185547,-1.4739127159118652,-4.087320804595947,-2.5942649841308594,-4.7366132736206055,0.4568643569946289,-1.5084915161132812,0.8927688598632812,-2.6819639205932617,1.2160263061523438,-0.1488351821899414,-0.14763784408569336,-0.7330856323242188,1.7434263229370117,-0.6915187835693359,0.6054115295410156,-0.6251468658447266,0.18656158447265625,-2.2440414428710938,-0.7320966720581055,0.9116287231445312,0.6675131916999817,0.6043910980224609,-0.0961465835571289,1.8600740432739258,1.4214582443237305,-1.205615520477295,-0.5177726745605469,0.15996170043945312,-0.2535896301269531,-2.2104239463806152,1.3427562713623047,0.041887521743774414,1.4897098541259766,0.060222506523132324,-0.34209203720092773,0.699284017086029,0.6842024326324463,0.5804225206375122,-1.1573050022125244,-5.029721736907959,-2.9460182189941406,-0.5197992324829102,-0.4073209762573242,-0.4258995056152344,0.09758925437927246,-0.17737483978271484,0.34000396728515625,-0.2322378158569336,-0.1475057601928711,0.3234586715698242,-4.021762371063232,1.0243587493896484,0.09681177139282227,-0.37044429779052734,-0.024251937866210938,-0.6807060241699219,-2.7765073776245117,-1.0258405208587646,-0.4300117492675781,-1.0900561809539795,0.01822376251220703,1.6176338195800781,0.41026878356933594,-1.8498859405517578,-1.1419553756713867,4.136728286743164,-2.150479793548584,3.3707027435302734,-0.1467914581298828,3.308709144592285,0.7113504409790039,-1.1078815460205078,-0.8801693916320801,-0.42049598693847656,0.09227943420410156,-0.9760036468505859,-0.24633145332336426,0.6273136138916016,-1.2083473205566406,2.211038112640381,1.31689453125,0.2544741630554199,-2.016566276550293,0.562079668045044,-0.21720147132873535,0.5428485870361328,2.6177520751953125,-0.41208529472351074,-2.123898506164551,1.1267423629760742,-2.7048110961914062,0.5039205551147461,2.575258255004883,-0.8799686431884766,0.87249755859375,0.6080217361450195,-1.5108757019042969,1.7290878295898438,-1.6399269104003906,-0.5428943634033203,0.84613037109375,-0.981050968170166,-0.1954975128173828,1.2940922975540161,0.1952195167541504,0.9415614604949951,0.2965078353881836,-0.3136320114135742,-1.9692754745483398,0.45517802238464355,0.7992758750915527,-0.8403658866882324,-1.3770065307617188,-0.8184165954589844,1.0756406784057617,-1.4569718837738037,0.12512969970703125,0.7871246337890625,0.5787014961242676,0.31616687774658203,-0.977025032043457,-0.012317657470703125,1.0037927627563477,0.013478279113769531,-0.30511474609375,-3.1156558990478516,-0.42231273651123047,-1.2464590072631836,-1.6617259979248047,-1.1023149490356445,1.036198616027832,-2.310985565185547,1.1404132843017578,-1.9509344100952148,0.5048604011535645,-1.2156801223754883,-1.9823888093233109,-0.035666465759277344,-0.6143108606338501,1.6386127471923828,0.5471515655517578,-1.9428901672363281,-0.4070549011230469,-2.773345947265625,-0.5113223791122437,1.6288013458251953,0.25351572036743164,0.15111017227172852,-0.3359489440917969,0.7992641925811768,2.072324752807617,2.300220489501953,-0.068145751953125,-0.9164752960205078,1.361440658569336,2.9279489517211914,-0.5041522979736328,-1.6832647323608398,2.254055976867676,0.4718012809753418,4.221163272857666,-2.008607864379883,2.1336441040039062,-0.046668052673339844,-0.021943092346191406,-0.45098376274108887,0.8991556167602539,0.2222919464111328,-0.8832874298095703,0.5224043130874634,-1.5545425415039062,-0.8891315460205078,-1.2283811569213867,-0.16979026794433594,-0.12035465240478516,-1.4396910667419434,2.688563823699951,0.3629484176635742,-0.3630373477935791,-0.8028202056884766,-0.817103385925293,2.4815568923950195,4.305469512939453,0.29107093811035156,-0.06596851348876953,-1.4165449142456055,-1.808523178100586,0.25835418701171875,3.450076103210449,2.4532294273376465,-1.139315128326416,-1.548055648803711,0.9842357635498047,2.5022711753845215,0.3585033416748047,-1.4773750305175781,4.402332782745361,-2.932069778442383,0.5611152648925781,-0.5744171142578125,-1.4430465698242188,-1.2840442657470703,-1.4089546203613281,-1.9141159057617188,-2.0861244201660156,-1.1009902954101562,-0.5784721374511719,0.5885696411132812,-2.700193405151367,-4.627159118652344,-2.626720428466797,0.15899276733398438,-3.0625839233398438,-1.1167106628417969,-0.9565067291259766,0.9147377014160156,-3.362211227416992,-1.9675369262695312,0.84918212890625,-0.3434715270996094,0.23302459716796875,0.3271751403808594,2.135133743286133,1.5063514709472656,-0.6943435668945312,-0.4322929382324219,-0.5549640655517578,-5.317935943603516,-2.768766403198242,2.094175338745117,-0.25445556640625,-1.0037117004394531,-0.44933128356933594,2.3595409393310547,-0.0001277923583984375,-2.2030487060546875,-1.2851371765136719,-2.4257545471191406,-0.23543930053710938,-5.91357421875,-0.10899162292480469,0.4516792297363281,-0.08846473693847656,3.5594730377197266,0.24981307983398438,-3.9659652709960938,-2.0962600708007812,-0.7912673950195312,-3.4756431579589844,0.1710186004638672,2.2766380310058594,2.130176544189453,-2.0331287384033203,-6.486848831176758,-1.7676429748535156,-1.4005661010742188,-1.4338607788085938,-7.2733306884765625,-0.21361541748046875,-3.5150184631347656,-2.126462936401367,0.9336280822753906,-0.3534126281738281,-1.1579360961914062,-1.6873798370361328,-3.991718292236328,-7.745372772216797,0.09248161315917969,0.20920372009277344,-2.0994510650634766,-0.6702499389648438,-0.8619899749755859,-2.541360855102539,-2.4034423828125,-6.483682632446289,1.5054874420166016,-0.4140472412109375,0.3016777038574219,-0.5571956634521484,0.4810011386871338,-0.3056058883666992,1.224477767944336,-1.5407676696777344,-0.6194171905517578,-1.1478376388549805,0.3519744873046875,0.3126211166381836,1.0338997840881348,-2.605215072631836,-2.494384765625,-0.2372746467590332,-0.5821037292480469,-0.576899528503418,-0.29944515228271484,-2.009650230407715,-1.3425159454345703,0.1277008056640625,-0.9794902801513672,0.026947021484375,-0.5780582427978516,-0.5129127502441406,0.9743156433105469,0.46813082695007324,2.4021129608154297,-1.5481414794921875,1.1262426376342773,-0.11989986896514893,3.92852783203125,-1.2881336212158203,-0.3455066680908203,0.08691787719726562,-1.417360782623291,-0.823521614074707,-1.198582410812378,-1.1853761672973633,0.43785810470581055,-0.5031461715698242,-0.13950061798095703,-5.617992401123047,0.13849687576293945,1.2886781692504883,-0.6611738204956055,-0.23154854774475098,-2.5401973724365234,0.5832557678222656,-0.26359081268310547,0.0,-1.8825807571411133,0.3294239044189453,-1.4768123626708984,-1.1305170059204102,-0.2191634178161621,-0.7228727340698242,-0.638923168182373,-0.9729480743408203,-0.6530570983886719,0.18054580688476562,1.6402597427368164,-2.513495922088623,-0.25154685974121094,-1.5316972732543945,-4.306976318359375,1.3398017883300781,0.039826393127441406]




def calculate_percentile(sortedarr, start, end):
    # generate dataset
    # dataset = 0.5 + rand(1000) * 0.5
    # bootstrap
    dataset = np.array(sortedarr)
    dataset = np.absolute(dataset)
    dataset = dataset[start:end]
    if len(dataset)==0:
        return []
    scores = list()

    for _ in range(100):
        # bootstrap sample
        indices = np.array(randint(0, len(dataset), len(dataset)))
        sample = dataset[indices]
        # calculate and store statistic

        statistic = mean(sample)
        scores.append(statistic)

    print('50th percentile (median) = %.3f' % median(scores))
    # calculate 95% confidence intervals (100 - alpha)
    alpha = 10.0
    # calculate lower percentile (e.g. 2.5)
    lower_p = alpha / 2.0
    upper_p = (100 - alpha) + (alpha / 2.0)
    # retrieve observation at lower percentile
    # lower = max(0.0, percentile(scores, lower_p))
    lower = min(percentile(scores, upper_p), percentile(scores, lower_p))
    print('%.1fth percentile = %.3f' % (lower_p, lower))
    # calculate upper percentile (e.g. 97.5)

    # retrieve observation at upper percentile

    upper = max(percentile(scores, lower_p), percentile(scores, upper_p))
    print('%.1fth percentile = %.3f' % (upper_p, upper))
    return [lower, upper]


def distribution_plot(data, data1=None, start=None, end=None, label="", figname=""):
    fig = plt.figure()
    fig.add_subplot(111)
    # sns_plot = sns.distplot(err_bin10, rug=True, hist=False)
    # for i in bin_list:
    #     sns.kdeplot(i)
    sns.kdeplot(data, label=label)
    plt.vlines(start, 0, 0.04, colors = "c", linestyles = "dashed", label='bin boundary')
    plt.vlines(end, 0, 0.04, colors = "c", linestyles = "dashed", label='bin boundary')
    data = np.sort(data)
    start_idx = np.searchsorted(data, start)
    end_idx = np.searchsorted(data, end)
    if start_idx<end_idx:
        [lower, upper] = calculate_percentile(data, start_idx, end_idx)
        plt.vlines(lower, 0, 0.04, colors = "r", linestyles = "dashed", label='Confidence lower bound')
        plt.vlines(upper, 0, 0.04, colors = "r", linestyles = "dashed", label='Confidence upper bound')
    # sns.kdeplot(data1, label="ground truth")
    plt.legend()
    # sns_plot = sns_plot.distplot(err_bin20, rug=True, hist=True)
    # sns_plot = sns_plot.get_figure()
    plt.savefig(figname)
    # sns_plot.savefig("bin10distribution.png")

def _get_error(dataset, start, end):
    dataset = np.absolute(dataset)
    mean_error = mean(dataset[start: end])
    return mean_error

def error_sliding_window(err, truth, binsize=10):
    dataset = np.array(err)
    # sorted_arr = np.sort(dataset)
    sorted_arr = truth
    mean_error_list = []
    percentile_list = []
    i = 0
    bin_list = []
    while i < int(max(sorted_arr)):
        # print(i)
        start_idx = np.searchsorted(truth, i)
        end_idx = np.searchsorted(truth, i+binsize)
        dataset = np.absolute(dataset)
        bin_list.append(dataset[start_idx: end_idx])
        mean_error_list.append([i, start_idx, end_idx, _get_error(dataset, start_idx, end_idx)])
        percentile_list.append([i, start_idx, end_idx]+ calculate_percentile(dataset, start_idx, end_idx))
        i += binsize
    # print(mean_error_list)
    # print(percentile_list)
    return mean_error_list, percentile_list, bin_list


def plot_upper_lower(x, upperarr, lowerarr):
    fig = plt.figure()
    fig.add_subplot(111)
    cat = np.concatenate((upperarr.reshape(-1,1), lowerarr.reshape(-1,1)), axis=1)
    # upperarr = pd.DataFrame(cat, columns=["upper", "lower"])

    # lowerarr = pd.DataFrame(lowerarr)

    # sns.lineplot(data=upperarr, color="coral")

    # sns.lineplot(data=lowerarr, color="coral", label="lower")

    # plt.legend()
    # sns_plot = sns_plot.distplot(err_bin20, rug=True, hist=True)
    # sns_plot = sns_plot.get_figure()
    plt.plot(x, upperarr, 'b-', label='upper')
    plt.plot(x, lowerarr, 'r-', label='lower')
    plt.savefig("upperlower.png")

def plot_line(x, y, label="", figname="upperlower.png"):
    fig = plt.figure()
    fig.add_subplot(111)
    # upperarr = pd.DataFrame(cat, columns=["upper", "lower"])

    # lowerarr = pd.DataFrame(lowerarr)

    # sns.lineplot(data=upperarr, color="coral")

    # sns.lineplot(data=lowerarr, color="coral", label="lower")

    # plt.legend()
    # sns_plot = sns_plot.distplot(err_bin20, rug=True, hist=True)
    # sns_plot = sns_plot.get_figure()
    plt.plot(x, y, 'b-', label=label)
    plt.savefig(figname)

def multibox_plot(data, figname="box.png"):
    fig7, ax7 = plt.subplots()
    ax7.set_title('Boxes with Different bins')
    ax7.boxplot(data)
    plt.savefig(figname)

def csv_converter():
    truth_list = []
    error_list = []
    predict_list = []
    for i in range(5):
        result_filename = "regress_bins_"+str(i)+"fold_VGG1550366626.18.csv"
        df = pd.read_csv("../result/"+result_filename, header=None)
        df.loc[1] = pd.to_numeric(df.loc[1])
        df = df.sort_values(by=1, axis=1)
        # print(df)
        truth_list.extend(df.loc[1])
        predict = df.loc[0].values
        predict = [float(i[1:-2]) for i in predict]
        predict = [0.0 if i<0 else i for i in predict ]
        predict_list.extend(predict)

        error = df.loc[2]
        error = [float(i[2:-2]) for i in error]
        error_list.extend(error)
    # print(truth_list)
    # print(predict_list)
    return truth_list, error_list, predict_list

if __name__ == '__main__':
    # distribution_plot()
    # result_filename = "regress_VGG1548019989.04_10.csv"
    binsize = 50
    truth_list, error_list, predict_list = csv_converter()


    total_df = pd.DataFrame([truth_list, error_list, predict_list])
    total_df = total_df.sort_values(by=2, axis=1)
    error_list = total_df.loc[1].values
    truth_list = total_df.loc[0].values
    predict_list = total_df.loc[2].values
    # print(error_list)
    # print(predict_list)

    mean_error_list, percentile_list, bin_list = error_sliding_window(truth_list, predict_list, binsize=binsize)
    print("*@"*30)
    print(bin_list)
    multibox_plot(bin_list)
    variance = [np.std(np.array(i)) for i in bin_list if len(i)>=2]
    x = [i*binsize for i in range(len(bin_list)) if len(bin_list[i])>=2]
    plot_line(x, variance, label="variance", figname="variance.png")

    distribution_plot(predict_list, truth_list, label="predict", figname="predict_distribution.png")

    for i in range(len(bin_list)):
        if len(bin_list[i])>=2:
            print(bin_list[i])
            distribution_plot(bin_list[i], label="bin"+str(i*binsize), start=i*binsize, end=(i+1)*binsize, figname="bin"+str(i*binsize)+".png" )

    # distribution_plot(bin_list, label="bin"+str(binsize), figname="total_bin"+str(binsize)+".png" )

    # mean_error_list =
    x = np.array([i[0] for i in percentile_list if len(i)>=4])
    upperarr = np.array([i[4] for i in percentile_list if len(i)>=4])
    lowerarr = np.array([i[3] for i in percentile_list if len(i)>=4])
    plot_upper_lower(x, upperarr, lowerarr)

    x = np.array([i[0] for i in mean_error_list if not np.isnan(i[3])])
    y = np.array([i[3] for i in mean_error_list if not np.isnan(i[3])])
    # print(x)
    # print(y)
    plot_line(x,y, label="mean error", figname="meanerror.png")
