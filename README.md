CAPM and Fama-French 3 Factors Model：
------

1.CAPM Model：
------
The Capital Asset Pricing Model (CAPM) was developed by American scholars William Sharpe, John Lintner, Jack Treynon and Jan Mossin in 1964 on the basis of portfolio theory and capital market theory, mainly to study the expectations of assets in the securities market and the relationship between returns and risky assets.

Formula：
------
E(return) = Rf + beta * (E(Market_return - Rf))，
that

E(return) is expected return， 

Rf is risk-free rate,（we use 5-year Chinese Government Bond here），

beta is sensitivity factors，

E(Market_return - Rf) is risk premium。

Example：
------
We test the one-year period from 2019 to 2020, take one week as the time unit sample period, the Shanghai Stock Exchange Index as the return benchmark, and the 5-year treasury bond interest rate as the risk-free return (weekly simple interest is obtained at the annual interest rate/52, (compound interest can also be considered, but here is for the convenience of calculation)).

The sample stocks are SSE 50, and their beta, R^2 and p_value are tested.

The results show that the more stable and traditional industry stocks (such as banking, insurance, heavy industry, etc.), the more effectively the stock price changes in the broader market can explain their stock price changes. Emerging industries are on the contrary, and the beta value is larger, indicating greater volatility than the benchmark.

Interestingly, there are two stocks, Changjiang Electric Power (CHINESE NAME:长江电力) and Haitian Flavor Industry (CHINESE NAME:海天味业) whose p-values are > .1, indicating that market fluctuations cannot explain the changes in their stock prices.
The reason is also obvious. Changjiang Electric Power is a long-term allocation of pension funds, while Haitian Flavor Industry is a stock under market manipulation.

          Name    Code    Beta       R^2            p_value
    0   包钢股份  600010  0.957129  0.462245   [2.973178706522945e-08]
    1   北方稀土  600111  0.799512  0.160111    [0.003290834159130382]
    2   恒力石化  600346  1.308105  0.265744   [9.194984838084904e-05]
    3   华友钴业  603799  1.361035  0.144010    [0.006567882521185264]
    4   中远海控  601919  1.737598  0.566359   [1.246840535780248e-10]
    5   长城汽车  601633  1.695117  0.320963  [1.1972313011463694e-05]
    6   长江电力  600900  0.192871  0.047607     [0.12020001761584796]
    7    片仔癀  600436  1.101855  0.365456   [2.074538822511093e-06]
    8   紫金矿业  601899  0.949219  0.230765   [0.0003602281583197898]
    9   航发动力  600893  0.902539  0.508455   [6.243141286791746e-09]
    10  山西汾酒  600809  1.038770  0.209492    [0.000645711637522729]
    11  通威股份  600438  1.283691  0.160498     [0.00324996823909229]
    12  兆易创新  603986  1.816992  0.244624  [0.00019375291463990304]
    13  韦尔股份  603501  1.899023  0.259560  [0.00011458356633056854]
    14  海天味业  603288  0.320605  0.046832     [0.12333649821751806]
    15  恒生电子  600570  1.839453  0.608059    [9.62961475843923e-12]
    16  闻泰科技  600745  2.138477  0.245112  [0.00019048604024831344]
    17  用友网络  600588  1.139941  0.197776   [0.0009560824531142106]
    18  隆基股份  601012  1.160645  0.198455   [0.0010532637666243242]
    19  中信建投  601066  2.458203  0.255630  [0.00013167844983226636]
    20  海通证券  600837  1.571875  0.431511   [1.231506280766989e-07]
    21  三一重工  600031  1.260254  0.333641   [7.344811322167318e-06]
    22  药明康德  603259  0.785547  0.136471    [0.007033881467069857]
    23  复星医药  600196  1.232031  0.389317   [7.731406773739181e-07]
    24  中国国旅  601888  0.847559  0.222333   [0.0004178992797805659]
    25  青岛海尔  600690  1.220898  0.382740  [1.0184469327442502e-06]
    26  恒瑞医药  600276  0.906641  0.314154  [1.5513081615372747e-05]
    27  海螺水泥  600585  1.355957  0.519630  [1.6765755008818614e-09]
    28  万华化学  600309  1.539063  0.437593   [9.350806611370228e-08]
    29  国泰君安  601211  1.255469  0.477217  [1.4462133653891912e-08]
    30  华泰证券  601688  1.643359  0.449716  [5.3553863525754036e-08]
    31  伊利股份  600887  0.923437  0.318470  [1.3167156911863209e-05]
    32  农业银行  601288  0.483984  0.344100   [4.877096926011006e-06]
    33  中国建筑  601668  0.929199  0.504025   [3.781571736454014e-09]
    34  中国太保  601601  1.415918  0.501630  [4.2751564611031424e-09]
    35  中国石油  601857  0.545996  0.469351  [2.1169632718789034e-08]
    36  中国神华  601088  1.098047  0.501050   [4.403631287976093e-09]
    37  中国平安  601318  1.314844  0.645563   [7.573867503341198e-13]
    38  兴业银行  601166  1.307715  0.476388  [1.5058425597629448e-08]
    39  中国人寿  601628  1.675586  0.487735   [8.613438980260887e-09]
    40  保利地产  600048  0.990039  0.339958  [5.7397910354180746e-06]
    41  工商银行  601398  0.605273  0.345287   [4.654053150310174e-06]
    42  贵州茅台  600519  0.872754  0.243664  [0.00020034825856854918]
    43  中信证券  600030  1.615625  0.634659  [2.7207859838933393e-12]
    44  上海汽车  600104  1.028223  0.331356   [8.025800734828799e-06]
    45  招商银行  600036  1.092383  0.540216   [5.508091912618415e-10]
    46  中国石化  600028  0.807520  0.507412  [3.1763453265317737e-09]


2.Fama-French 3 Factors Model：
------
The FF3 factor model is derived from the CAPM model and the APT theory.
In 1992, Fama and French studied the factors that determine the difference in the return of different stocks in the US stock market and found that the beta value of the stock market cannot explain the difference in the return of different stocks,
The market value, book-to-market ratio, and price-earnings ratio of listed companies can explain the differences in stock returns.
Fama and French believe that the above excess returns are compensation for risk factors that are not reflected in the beta in the CAPM.

But in the Chinese market, the SMB and HML factors have failed to shine. Liu et al. (2018) pointed out that "shell effect"/"shell pollution" is the main reason for the failure of explanation.
Therefore, I re-adjusted SMB and HML for the Chinese market, removed the companies with the last 1/3 market value, and only kept the top 2/3 companies for SMB and HML calculations.

Liu, J., R. F. Stambaugh, and Y. Yuan (2018). Size and Value in China. Journal of Financial Economics, forthcoming.

Formula：
------
Rp - Rf = a + b1 * RMRF + b2 * SMB + b3 * HML + error

which：

Rp is stock return，

Rf is risk-free return,

bi is sensitivity factor

RMRF is market risk premium

a is intercept

e is error

Example：
------
We examine the one-year period from 2019 to 2020, take one week as the time unit sample period, the Shanghai Stock Exchange Index as the return benchmark, and the 5-year treasury bond interest rate as the risk-free return (weekly simple interest is obtained at the annual interest rate/52, (compound interest can also be considered, but here is still using the simple rate for the convenience of calculation)).

The sample stocks are SSE 50, and their utility for the three-factor model is tested.

The results are undoubtedly instructive: we found that the traditional CAPM model can better explain the stock price changes, after adding the SMB and HML factors,
Interpretation ability has been greatly enhanced.
For example: Industrial Bank(Chinese Name:兴业银行) R^2 0.47 -> 0.77.

For stocks that cannot easily explain stock price changes with CAPM, adding factors is a burden.
Such as: Northern rare earth(Chinese Name：北方稀土) R^2 0.16 -> 0.17. And neither SMB nor HML passed the hypothesis test.

Therefore, we draw a preliminary conclusion that the FF3 model has certain guiding significance in the Shanghai Stock Exchange 50, but the significance remains in a company that is stable enough and has no major changes in the development of the industry.


     Unnamed: 0      Name     Code   R^2  f_p_value  coeff_constant  p_value_constant  coeff_market  p_value_Market  coeff_SMB  p_value_SMB  coeff_HML  p_value_HML
    0            0  包钢股份  600010 0.609      0.000           0.017             0.961         0.000           0.000      0.001        0.001      0.031        0.031
    1            1  北方稀土  600111 0.177      0.026           0.469             0.518         0.011           0.011      0.257        0.257      0.805        0.805
    2            2  恒力石化  600346 0.301      0.001           1.055             0.209         0.000           0.000      0.104        0.104      0.725        0.725
    3            3  华友钴业  603799 0.187      0.024           0.387             0.769         0.004           0.004      0.469        0.469      0.173        0.173
    4            4  中远海控  601919 0.569      0.000           0.211             0.727         0.000           0.000      0.703        0.703      0.484        0.484
    5            5  长城汽车  601633 0.323      0.000           0.657             0.500         0.000           0.000      0.329        0.329      0.953        0.953
    6            6  长江电力  600900 0.042      0.563           0.296             0.390         0.161           0.161      0.969        0.969      0.736        0.736
    7            7   片仔癀  600436 0.510      0.000          -0.729             0.157         0.000           0.000      0.047        0.047      0.001        0.001
    8            8  紫金矿业  601899 0.242      0.005           0.175             0.798         0.000           0.000      0.473        0.473      0.248        0.248
    9            9  航发动力  600893 0.613      0.000          -0.445             0.184         0.000           0.000      0.010        0.010      0.399        0.399
    10          10  山西汾酒  600809 0.454      0.000           0.321             0.610         0.000           0.000      0.016        0.016      0.000        0.000
    11          11  通威股份  600438 0.218      0.009          -0.129             0.908         0.001           0.001      0.473        0.473      0.050        0.050
    12          12  兆易创新  603986 0.267      0.002           1.893             0.121         0.000           0.000      0.307        0.307      0.374        0.374
    13          13  韦尔股份  603501 0.391      0.000           2.259             0.040         0.000           0.000      0.067        0.067      0.028        0.028
    14          14  海天味业  603288 0.333      0.000          -0.112             0.812         0.004           0.004      0.007        0.007      0.000        0.000
    15          15  恒生电子  600570 0.627      0.000           0.509             0.367         0.000           0.000      0.095        0.095      0.445        0.445
    16          16  闻泰科技  600745 0.246      0.004           2.537             0.081         0.001           0.001      0.470        0.470      0.614        0.614
    17          17  用友网络  600588 0.367      0.000           1.023             0.206         0.001           0.001      0.002        0.002      0.399        0.399
    18          18  隆基股份  601012 0.279      0.002          -0.107             0.904         0.000           0.000      0.152        0.152      0.027        0.027
    19          19  中信建投  601066 0.287      0.001           3.504             0.032         0.002           0.002      0.164        0.164      0.199        0.199
    20          20  海通证券  600837 0.481      0.000           1.456             0.036         0.000           0.000      0.103        0.103      0.055        0.055
    21          21  三一重工  600031 0.332      0.000           1.187             0.085         0.000           0.000      0.452        0.452      0.714        0.714
    22          22  药明康德  603259 0.300      0.001          -0.147             0.833         0.000           0.000      0.221        0.221      0.001        0.001
    23          23  复星医药  600196 0.552      0.000          -1.055             0.051         0.000           0.000      0.011        0.011      0.001        0.001
    24          24  中国国旅  601888 0.439      0.000          -0.467             0.381         0.000           0.000      0.217        0.217      0.000        0.000
    25          25  青岛海尔  600690 0.588      0.000          -0.070             0.890         0.000           0.000      0.000        0.000      0.411        0.411
    26          26  恒瑞医药  600276 0.544      0.000           0.128             0.757         0.000           0.000      0.004        0.004      0.000        0.000
    27          27  海螺水泥  600585 0.598      0.000           0.556             0.225         0.000           0.000      0.007        0.007      0.095        0.095
    28          28  万华化学  600309 0.633      0.000           0.449             0.404         0.000           0.000      0.000        0.000      0.025        0.025
    29          29  国泰君安  601211 0.498      0.000           0.357             0.490         0.000           0.000      0.787        0.787      0.139        0.139
    30          30  华泰证券  601688 0.501      0.000           0.282             0.685         0.000           0.000      0.149        0.149      0.168        0.168
    31          31  伊利股份  600887 0.367      0.000           0.046             0.929         0.000           0.000      0.065        0.065      0.178        0.178
    32          32  农业银行  601288 0.777      0.000           0.338             0.035         0.000           0.000      0.000        0.000      0.000        0.000
    33          33  中国建筑  601668 0.636      0.000           0.119             0.708         0.000           0.000      0.251        0.251      0.001        0.001
    34          34  中国太保  601601 0.611      0.000           0.166             0.738         0.000           0.000      0.001        0.001      0.486        0.486
    35          35  中国石油  601857 0.573      0.000          -0.463             0.027         0.000           0.000      0.596        0.596      0.198        0.198
    36          36  中国神华  601088 0.580      0.000           0.113             0.780         0.000           0.000      0.223        0.223      0.026        0.026
    37          37  中国平安  601318 0.753      0.000           0.222             0.485         0.000           0.000      0.000        0.000      0.271        0.271
    38          38  兴业银行  601166 0.774      0.000           0.494             0.171         0.000           0.000      0.000        0.000      0.005        0.005
    39          39  中国人寿  601628 0.678      0.000           0.502             0.351         0.000           0.000      0.000        0.000      0.345        0.345
    40          40  保利地产  600048 0.455      0.000           1.059             0.037         0.000           0.000      0.388        0.388      0.005        0.005
    41          41  工商银行  601398 0.739      0.000           0.474             0.028         0.000           0.000      0.000        0.000      0.000        0.000
    42          42  贵州茅台  600519 0.570      0.000           0.040             0.926         0.000           0.000      0.000        0.000      0.000        0.000
    43          43  中信证券  600030 0.647      0.000           0.302             0.542         0.000           0.000      0.166        0.166      0.737        0.737
    44          44  上海汽车  600104 0.435      0.000          -0.642             0.238         0.000           0.000      0.008        0.008      0.865        0.865
    45          45  招商银行  600036 0.676      0.000           0.565             0.089         0.000           0.000      0.000        0.000      0.453        0.453
    46          46  中国石化  600028 0.603      0.000           0.119             0.678         0.000           0.000      0.041        0.041      0.038        0.038

（Chinese Version)
CAPM与Fama-French 3因子模型：
------

1.CAPM模型：
------
资本资产定价模型（Capital Asset Pricing Model 简称CAPM）是由美国学者威廉·夏普（William Sharpe）、林特尔（John Lintner）、特里诺（Jack Treynor）和莫辛（Jan Mossin）等人于1964年在资产组合理论和资本市场理论的基础上发展起来的，主要研究证券市场中资产的预期收益率与风险资产之间的关系.

具体公式：
------
E(return) = Rf + beta * (E(Market_return - Rf))，
其中： 

E(return) 为期望回报， 

Rf为无风险收益率（我们采用5年期中国国债），

beta为敏感系数，

E(Market_return - Rf) 为风险溢价。

举例：
------
我们检验2019年-2020年一年时间，以一周为时间单位样本周期，以上证指数为回报基准，5年期国债利率为无风险回报（以年利率/52得到每周单利，（亦可考虑复利，此处为计算方便））。

样本股为上证50，检验他们的beta， R^2 和 p_value。

结果显示越是稳定且传统行业的股票（如银行，保险，重工业等），大盘的股价变动越是能有效解释其股价变化。新兴产业则反之，且beta值较大，说明相比基准波动更大。

有趣的是，有两个股票"长江电力"和"海天味业"的p—value > .1,说明大盘波动无法解释两者股价变化。
道理也是显然的，长江电力是养老基金长久配置票，而海天味业则是公认的庄股，显而易见。


        品种名称   代码    Beta       R^2            p_value
    0   包钢股份  600010  0.957129  0.462245   [2.973178706522945e-08]
    1   北方稀土  600111  0.799512  0.160111    [0.003290834159130382]
    2   恒力石化  600346  1.308105  0.265744   [9.194984838084904e-05]
    3   华友钴业  603799  1.361035  0.144010    [0.006567882521185264]
    4   中远海控  601919  1.737598  0.566359   [1.246840535780248e-10]
    5   长城汽车  601633  1.695117  0.320963  [1.1972313011463694e-05]
    6   长江电力  600900  0.192871  0.047607     [0.12020001761584796]
    7    片仔癀  600436  1.101855  0.365456   [2.074538822511093e-06]
    8   紫金矿业  601899  0.949219  0.230765   [0.0003602281583197898]
    9   航发动力  600893  0.902539  0.508455   [6.243141286791746e-09]
    10  山西汾酒  600809  1.038770  0.209492    [0.000645711637522729]
    11  通威股份  600438  1.283691  0.160498     [0.00324996823909229]
    12  兆易创新  603986  1.816992  0.244624  [0.00019375291463990304]
    13  韦尔股份  603501  1.899023  0.259560  [0.00011458356633056854]
    14  海天味业  603288  0.320605  0.046832     [0.12333649821751806]
    15  恒生电子  600570  1.839453  0.608059    [9.62961475843923e-12]
    16  闻泰科技  600745  2.138477  0.245112  [0.00019048604024831344]
    17  用友网络  600588  1.139941  0.197776   [0.0009560824531142106]
    18  隆基股份  601012  1.160645  0.198455   [0.0010532637666243242]
    19  中信建投  601066  2.458203  0.255630  [0.00013167844983226636]
    20  海通证券  600837  1.571875  0.431511   [1.231506280766989e-07]
    21  三一重工  600031  1.260254  0.333641   [7.344811322167318e-06]
    22  药明康德  603259  0.785547  0.136471    [0.007033881467069857]
    23  复星医药  600196  1.232031  0.389317   [7.731406773739181e-07]
    24  中国国旅  601888  0.847559  0.222333   [0.0004178992797805659]
    25  青岛海尔  600690  1.220898  0.382740  [1.0184469327442502e-06]
    26  恒瑞医药  600276  0.906641  0.314154  [1.5513081615372747e-05]
    27  海螺水泥  600585  1.355957  0.519630  [1.6765755008818614e-09]
    28  万华化学  600309  1.539063  0.437593   [9.350806611370228e-08]
    29  国泰君安  601211  1.255469  0.477217  [1.4462133653891912e-08]
    30  华泰证券  601688  1.643359  0.449716  [5.3553863525754036e-08]
    31  伊利股份  600887  0.923437  0.318470  [1.3167156911863209e-05]
    32  农业银行  601288  0.483984  0.344100   [4.877096926011006e-06]
    33  中国建筑  601668  0.929199  0.504025   [3.781571736454014e-09]
    34  中国太保  601601  1.415918  0.501630  [4.2751564611031424e-09]
    35  中国石油  601857  0.545996  0.469351  [2.1169632718789034e-08]
    36  中国神华  601088  1.098047  0.501050   [4.403631287976093e-09]
    37  中国平安  601318  1.314844  0.645563   [7.573867503341198e-13]
    38  兴业银行  601166  1.307715  0.476388  [1.5058425597629448e-08]
    39  中国人寿  601628  1.675586  0.487735   [8.613438980260887e-09]
    40  保利地产  600048  0.990039  0.339958  [5.7397910354180746e-06]
    41  工商银行  601398  0.605273  0.345287   [4.654053150310174e-06]
    42  贵州茅台  600519  0.872754  0.243664  [0.00020034825856854918]
    43  中信证券  600030  1.615625  0.634659  [2.7207859838933393e-12]
    44  上海汽车  600104  1.028223  0.331356   [8.025800734828799e-06]
    45  招商银行  600036  1.092383  0.540216   [5.508091912618415e-10]
    46  中国石化  600028  0.807520  0.507412  [3.1763453265317737e-09]


2.Fama-French 3因子模型：
------
FF3因子模型由CAPM模型 和 APT 理论 衍生而来。
1992年，Fama和French对美国股票市场决定不同股票回报率差异的因素的研究发现，股票的市场的beta值不能解释不同股票回报率的差异，
而上市公司的市值、账面市值比、市盈率可以解释股票回报率的差异。
Fama and French 认为，上述超额收益是对CAPM中β未能反映的风险因素的补偿。

但在中国市场，SMB 和 HML 因子并未能大放异彩。Liu et al. (2018) 则指出"壳效应"/"壳污染"是解释失效的主要原因。
因此我针对中国市场对SMB 和 HML 重新进行调整，去除后1/3市值的公司，只保留前2/3市值的公司做SMB 和 HML 计算。

Liu, J., R. F. Stambaugh, and Y. Yuan (2018). Size and Value in China. Journal of Financial Economics, forthcoming.

具体公式：
------
Rp - Rf = a + b1 * RMRF + b2 * SMB + b3 * HML + error

其中：

Rp 为 股票收益，

Rf 为 无风险收益

bi 为 敏感系数

RMRF 为 市场风险溢价

a 为 截距

e 为 误差

举例：
------
我们检验2019年-2020年一年时间，以一周为时间单位样本周期，以上证指数为回报基准，5年期国债利率为无风险回报（以年利率/52得到每周单利，（亦可考虑复利，此处为计算方便））。

样本股为上证50，检验他们对于三因子模型的效用。

结果无疑是具有指导意义的：我们发现，传统CAPM模型能较好解释股价变动的股票，在加入SMB 和 HML 因子后，
解释能力得到大幅度增强。
如：兴业银行 R^2 0.47 -> 0.77。

而对于不能较好用CAPM解释股价变动的股票而言，加入因子后反而是累赘。
如：北方稀土 R^2 0.16 -> 0.17。而且SMB 和 HML 均未通过假设检验。

因此，我们得出初步结论，FF3模型在上证50具有一定指导意义，但意义停留在足够稳定，行业发展无较大变化的公司。


     Unnamed: 0     品种名称    代码   R^2  f_p_value  coeff_constant  p_value_constant  coeff_market  p_value_Market  coeff_SMB  p_value_SMB  coeff_HML  p_value_HML
    0            0  包钢股份  600010 0.609      0.000           0.017             0.961         0.000           0.000      0.001        0.001      0.031        0.031
    1            1  北方稀土  600111 0.177      0.026           0.469             0.518         0.011           0.011      0.257        0.257      0.805        0.805
    2            2  恒力石化  600346 0.301      0.001           1.055             0.209         0.000           0.000      0.104        0.104      0.725        0.725
    3            3  华友钴业  603799 0.187      0.024           0.387             0.769         0.004           0.004      0.469        0.469      0.173        0.173
    4            4  中远海控  601919 0.569      0.000           0.211             0.727         0.000           0.000      0.703        0.703      0.484        0.484
    5            5  长城汽车  601633 0.323      0.000           0.657             0.500         0.000           0.000      0.329        0.329      0.953        0.953
    6            6  长江电力  600900 0.042      0.563           0.296             0.390         0.161           0.161      0.969        0.969      0.736        0.736
    7            7   片仔癀  600436 0.510      0.000          -0.729             0.157         0.000           0.000      0.047        0.047      0.001        0.001
    8            8  紫金矿业  601899 0.242      0.005           0.175             0.798         0.000           0.000      0.473        0.473      0.248        0.248
    9            9  航发动力  600893 0.613      0.000          -0.445             0.184         0.000           0.000      0.010        0.010      0.399        0.399
    10          10  山西汾酒  600809 0.454      0.000           0.321             0.610         0.000           0.000      0.016        0.016      0.000        0.000
    11          11  通威股份  600438 0.218      0.009          -0.129             0.908         0.001           0.001      0.473        0.473      0.050        0.050
    12          12  兆易创新  603986 0.267      0.002           1.893             0.121         0.000           0.000      0.307        0.307      0.374        0.374
    13          13  韦尔股份  603501 0.391      0.000           2.259             0.040         0.000           0.000      0.067        0.067      0.028        0.028
    14          14  海天味业  603288 0.333      0.000          -0.112             0.812         0.004           0.004      0.007        0.007      0.000        0.000
    15          15  恒生电子  600570 0.627      0.000           0.509             0.367         0.000           0.000      0.095        0.095      0.445        0.445
    16          16  闻泰科技  600745 0.246      0.004           2.537             0.081         0.001           0.001      0.470        0.470      0.614        0.614
    17          17  用友网络  600588 0.367      0.000           1.023             0.206         0.001           0.001      0.002        0.002      0.399        0.399
    18          18  隆基股份  601012 0.279      0.002          -0.107             0.904         0.000           0.000      0.152        0.152      0.027        0.027
    19          19  中信建投  601066 0.287      0.001           3.504             0.032         0.002           0.002      0.164        0.164      0.199        0.199
    20          20  海通证券  600837 0.481      0.000           1.456             0.036         0.000           0.000      0.103        0.103      0.055        0.055
    21          21  三一重工  600031 0.332      0.000           1.187             0.085         0.000           0.000      0.452        0.452      0.714        0.714
    22          22  药明康德  603259 0.300      0.001          -0.147             0.833         0.000           0.000      0.221        0.221      0.001        0.001
    23          23  复星医药  600196 0.552      0.000          -1.055             0.051         0.000           0.000      0.011        0.011      0.001        0.001
    24          24  中国国旅  601888 0.439      0.000          -0.467             0.381         0.000           0.000      0.217        0.217      0.000        0.000
    25          25  青岛海尔  600690 0.588      0.000          -0.070             0.890         0.000           0.000      0.000        0.000      0.411        0.411
    26          26  恒瑞医药  600276 0.544      0.000           0.128             0.757         0.000           0.000      0.004        0.004      0.000        0.000
    27          27  海螺水泥  600585 0.598      0.000           0.556             0.225         0.000           0.000      0.007        0.007      0.095        0.095
    28          28  万华化学  600309 0.633      0.000           0.449             0.404         0.000           0.000      0.000        0.000      0.025        0.025
    29          29  国泰君安  601211 0.498      0.000           0.357             0.490         0.000           0.000      0.787        0.787      0.139        0.139
    30          30  华泰证券  601688 0.501      0.000           0.282             0.685         0.000           0.000      0.149        0.149      0.168        0.168
    31          31  伊利股份  600887 0.367      0.000           0.046             0.929         0.000           0.000      0.065        0.065      0.178        0.178
    32          32  农业银行  601288 0.777      0.000           0.338             0.035         0.000           0.000      0.000        0.000      0.000        0.000
    33          33  中国建筑  601668 0.636      0.000           0.119             0.708         0.000           0.000      0.251        0.251      0.001        0.001
    34          34  中国太保  601601 0.611      0.000           0.166             0.738         0.000           0.000      0.001        0.001      0.486        0.486
    35          35  中国石油  601857 0.573      0.000          -0.463             0.027         0.000           0.000      0.596        0.596      0.198        0.198
    36          36  中国神华  601088 0.580      0.000           0.113             0.780         0.000           0.000      0.223        0.223      0.026        0.026
    37          37  中国平安  601318 0.753      0.000           0.222             0.485         0.000           0.000      0.000        0.000      0.271        0.271
    38          38  兴业银行  601166 0.774      0.000           0.494             0.171         0.000           0.000      0.000        0.000      0.005        0.005
    39          39  中国人寿  601628 0.678      0.000           0.502             0.351         0.000           0.000      0.000        0.000      0.345        0.345
    40          40  保利地产  600048 0.455      0.000           1.059             0.037         0.000           0.000      0.388        0.388      0.005        0.005
    41          41  工商银行  601398 0.739      0.000           0.474             0.028         0.000           0.000      0.000        0.000      0.000        0.000
    42          42  贵州茅台  600519 0.570      0.000           0.040             0.926         0.000           0.000      0.000        0.000      0.000        0.000
    43          43  中信证券  600030 0.647      0.000           0.302             0.542         0.000           0.000      0.166        0.166      0.737        0.737
    44          44  上海汽车  600104 0.435      0.000          -0.642             0.238         0.000           0.000      0.008        0.008      0.865        0.865
    45          45  招商银行  600036 0.676      0.000           0.565             0.089         0.000           0.000      0.000        0.000      0.453        0.453
    46          46  中国石化  600028 0.603      0.000           0.119             0.678         0.000           0.000      0.041        0.041      0.038        0.038
