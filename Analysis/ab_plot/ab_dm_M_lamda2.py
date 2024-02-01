import matplotlib.pyplot as plt

M_3_lamda0001 = [7.615819931030273, 7.5395890235900875, 7.396581649780273, 7.2455991268157955, 7.084253168106079, 6.910071992874146, 6.7206378936767575, 6.513669872283936, 6.2871287822723385, 6.039359855651855, 5.769372415542603, 5.476994466781616, 5.162857294082642, 4.828953409194947, 4.478893136978149, 4.117981910705566, 3.7529729843139648, 3.3917206525802612, 3.042691683769226, 2.7138325691223146, 2.411798119544983, 2.14137225151062, 1.9050933957099914, 1.7034000873565673, 1.5348266005516051, 1.3964277029037475, 1.284516978263855, 1.1951629996299744, 1.1245326757431031, 1.0691302418708801, 1.0259112000465394, 0.9923218488693237, 0.9662690043449402, 0.946074378490448, 0.9304107427597046, 0.9182233452796936, 0.9087107002735137, 0.901238638162613, 0.8953235924243927, 0.8905919909477233, 0.8867575764656067, 0.8835987567901611, 0.8809464991092681, 0.8786700010299683, 0.8766673266887665, 0.8748674333095551, 0.8731962025165558, 0.8716001451015473, 0.8700539588928222, 0.8685224115848541, 0.8669755816459656, 0.8653904139995575, 0.8637451350688934, 0.8620199918746948, 0.8601950168609619, 0.8582501947879791, 0.856164014339447, 0.8539139151573181, 0.8514731705188752, 0.8488228857517243, 0.8459144711494446, 0.8427031338214874, 0.8391631066799163, 0.8352344453334808, 0.8308531880378723, 0.825943398475647, 0.8204148352146149, 0.8141598284244538, 0.8070506989955902, 0.7989364206790924, 0.7896399438381195, 0.7789602041244507, 0.7666761934757232, 0.7525445222854614, 0.7363499939441681, 0.7179090857505799, 0.6971316158771514, 0.6740745007991791, 0.6489935338497161, 0.622359836101532, 0.5948149144649506, 0.5670580089092254, 0.5397093951702118, 0.5132183909416199, 0.48785490691661837, 0.46375989019870756, 0.441005602478981, 0.4196248292922974, 0.39962070882320405, 0.3809671252965927, 0.36361473202705386, 0.34749704599380493, 0.3325386315584183, 0.318660232424736, 0.3057830274105072, 0.2938303083181381, 0.2827293336391449, 0.2724114000797272, 0.26281302273273466, 0.2538755238056183, 0.24554461538791655, 0.23777071088552476, 0.2305084392428398, 0.22371671944856644, 0.21735780835151672, 0.21139746457338332, 0.20580423027276992, 0.20054963380098342, 0.19560801833868027, 0.1909553349018097, 0.18657009303569794, 0.18243279159069062, 0.178525647521019, 0.17483177185058593, 0.17133652567863464, 0.16802580505609513, 0.16488714665174484, 0.1619087278842926, 0.15908032655715942, 0.15639184266328812, 0.15383437126874924, 0.15139966905117036, 0.14907983541488648, 0.14686772376298904, 0.144757080078125, 0.1427416607737541, 0.14081588983535767, 0.1389742910861969, 0.1372123181819916, 0.13552550375461578, 0.13390948325395585, 0.13236031234264373, 0.13087433576583862, 0.12944821864366532, 0.128078830242157, 0.1267631769180298, 0.1254984676837921, 0.12428204491734504, 0.12311160191893578, 0.12198478356003761, 0.12089934051036835, 0.11985338479280472, 0.11884509548544883, 0.11787257492542266, 0.11693416088819504, 0.11602824851870537, 0.11515336632728576, 0.11430817097425461, 0.11349129378795624, 0.11270152255892754, 0.11193767562508583, 0.11119852289557457, 0.11048303619027137, 0.1097902812063694, 0.10911926180124283, 0.10846911370754242, 0.10783893465995789, 0.10722792223095894, 0.10663533210754395, 0.10606039911508561, 0.10550234466791153, 0.10496078133583069, 0.10443476289510727, 0.10392389371991158, 0.10342752709984779, 0.10294507890939712, 0.10247616097331047, 0.10202009528875351, 0.10157653465867042, 0.10114504992961884, 0.10072508230805396, 0.10031635761260986, 0.09991842433810234, 0.09953086450695992, 0.09915339425206185, 0.09878559410572052, 0.09842718169093131, 0.0980778269469738, 0.09773720353841782, 0.09740506932139396, 0.09708109721541405, 0.09676504358649254, 0.09645669087767601, 0.09615569040179253, 0.09586194008588791, 0.09557510018348694, 0.09529496654868126, 0.09502140805125237, 0.09475412592291832, 0.09449301883578301, 0.09423786103725433, 0.09398848935961723, 0.09374468550086021, 0.0935063637793064, 0.09327316135168076, 0.09304514452815056, 0.09282207489013672, 0.09260379076004029, 0.0923902340233326, 0.09218124374747276, 0.0919765755534172]
M_5_lamda0001 = [8.2810640335083, 8.269305515289307, 8.246989345550537, 8.222937679290771, 8.196534633636475, 8.167090511322021, 8.133787250518798, 8.09567632675171, 8.051623153686524, 8.000267696380615, 7.939954614639282, 7.868678617477417, 7.784008264541626, 7.683017683029175, 7.562236261367798, 7.4175652980804445, 7.244344234466553, 7.037428617477417, 6.791485118865967, 6.501540803909302, 6.163762187957763, 5.7765813827514645, 5.341436529159546, 4.864899826049805, 4.359356927871704, 3.842355227470398, 3.33537871837616, 2.86062433719635, 2.4366852283477782, 2.0755907893180847, 1.781309223175049, 1.5506340861320496, 1.3756208300590516, 1.2462403178215027, 1.152386224269867, 1.0851466417312623, 1.0373340129852295, 1.003454613685608, 0.9794586539268494, 0.962429803609848, 0.9502980768680572, 0.9416079938411712, 0.9353401839733124, 0.9307804703712463, 0.9274297654628754, 0.9249423563480377, 0.9230678498744964, 0.9216197729110718, 0.9204905509948731, 0.9195893049240113, 0.9188514292240143, 0.9182319283485413, 0.9176977574825287, 0.9172246754169464, 0.9167930364608765, 0.9163943350315094, 0.9160163044929505, 0.9156318783760071, 0.9152556002140045, 0.914876651763916, 0.9144888818264008, 0.9140843212604522, 0.913659644126892, 0.9132096111774445, 0.9127272486686706, 0.9122165262699127, 0.9116447567939758, 0.9110184788703919, 0.9103297829627991, 0.9095622181892395, 0.9086994171142578, 0.9077198743820191, 0.906597501039505, 0.9052976727485657, 0.9037791430950165, 0.9019858360290527, 0.8998118162155151, 0.8971793532371521, 0.8939441680908203, 0.889926016330719, 0.8849084079265594, 0.8786447048187256, 0.8708916187286377, 0.8614016652107239, 0.8497534334659577, 0.8348564803600311, 0.8144986510276795, 0.786072188615799, 0.7485326111316681, 0.703690767288208, 0.6562124252319336, 0.6116619288921357, 0.571676903963089, 0.534563934803009, 0.500115442276001, 0.46902658641338346, 0.44127863049507143, 0.41654691100120544, 0.3945066690444946, 0.37483066916465757, 0.35723025500774386, 0.34145220220088957, 0.3272725373506546, 0.31449371576309204, 0.3029445469379425, 0.29247707724571226, 0.2829641789197922, 0.2742957234382629, 0.266375920176506, 0.25912233591079714, 0.25246268808841704, 0.24633408784866334, 0.2406816601753235, 0.23545738756656648, 0.23061903566122055, 0.22612936198711395, 0.22195567041635514, 0.21806879937648774, 0.21444268375635148, 0.21105437129735946, 0.20788350999355315, 0.20491145849227904, 0.20212187319993974, 0.1995001643896103, 0.19703265130519867, 0.19470769613981248, 0.19251428544521332, 0.19044254422187806, 0.1884834885597229, 0.186628994345665, 0.1848717674612999, 0.1832050070166588, 0.1816224202513695, 0.18011850118637085, 0.17868807017803193, 0.1773262783885002, 0.17602878659963608, 0.17479155510663985, 0.17361082881689072, 0.1724833533167839, 0.1714056134223938, 0.17037499099969863, 0.16938842087984085, 0.16844374388456346, 0.1675383746623993, 0.166670323908329, 0.16583736389875411, 0.16503758430480958, 0.16426931470632553, 0.16353090107440948, 0.16282077580690385, 0.1621374249458313, 0.16147967129945756, 0.16084599494934082, 0.16023533791303635, 0.15964661538600922, 0.15907871574163437, 0.15853074193000793, 0.15800163745880128, 0.15749060064554216, 0.15699676871299745, 0.15651935636997222, 0.1560576841235161, 0.15561098158359526, 0.1551786780357361, 0.15476006865501404, 0.15435465127229692, 0.1539618492126465, 0.15358105897903443, 0.15321175754070282, 0.1528535842895508, 0.15250604897737502, 0.15216872543096543, 0.15184113532304763, 0.1515229970216751, 0.15121394991874695, 0.1509135454893112, 0.15062153190374375, 0.150337515771389, 0.15006128549575806, 0.14979251623153686, 0.14953087270259857, 0.14927616268396376, 0.14902810901403427, 0.14878649413585662, 0.14855112582445146, 0.14832166582345963, 0.1480979949235916, 0.1478799045085907, 0.14766716361045837, 0.14745966643095015, 0.14725714325904846, 0.14705951362848282, 0.14686655402183532, 0.1466781511902809, 0.14649408012628556, 0.1463142603635788, 0.1461385264992714, 0.14596680104732512, 0.14579889923334122, 0.14563477635383607]
M_10_lamda0001 = [8.392271995544434, 8.392195415496825, 8.392059326171875, 8.391919231414795, 8.391764450073243, 8.391589069366455, 8.39138650894165, 8.391150283813477, 8.39087200164795, 8.390540313720702, 8.390143394470215, 8.389661979675292, 8.389074802398682, 8.388352298736573, 8.387454605102539, 8.386332511901855, 8.384916210174561, 8.38311834335327, 8.380818367004395, 8.377860260009765, 8.374031734466552, 8.369039344787598, 8.362498378753662, 8.353892803192139, 8.342509555816651, 8.327394199371337, 8.307244396209716, 8.280303287506104, 8.244200706481934, 8.195762729644775, 8.130792045593262, 8.043858814239503, 7.928147220611573, 7.775548124313355, 7.577264356613159, 7.325409364700318, 7.016155481338501, 6.654768610000611, 6.261732196807861, 5.876347398757934, 5.550848531723022, 5.329039525985718, 5.218097162246704, 5.182955741882324, 5.177788496017456, 5.177441883087158, 5.176425981521606, 5.17488842010498, 5.172541570663452, 5.168661308288574, 5.1626122951507565, 5.153950786590576, 5.142541885375977, 5.128578519821167, 5.112077808380127, 5.094533681869507, 5.080173778533935, 5.069477319717407, 5.0565026760101315, 5.038724088668824, 5.01606068611145, 4.988003587722778, 4.951754426956176, 4.898375797271728, 4.818204736709594, 4.699405384063721, 4.526754713058471, 4.287472534179687, 3.977024531364441, 3.609995174407959, 3.2306549310684205, 2.9017590284347534, 2.6690119981765745, 2.5362252473831175, 2.4753822326660155, 2.45114324092865, 2.4388010025024416, 2.4262082099914553, 2.407360076904297, 2.3773184537887575, 2.3297462940216063, 2.2562376499176025, 2.1473190546035767, 1.9962345600128173, 1.8063652157783507, 1.598542833328247, 1.4071183681488038, 1.2595255374908447, 1.1600266337394713, 1.0974642634391785, 1.059726631641388, 1.0378959774971008, 1.0256170153617858, 1.0187211275100707, 1.014781653881073, 1.0124754428863525, 1.0110849618911744, 1.0102182745933532, 1.0096601247787476, 1.009290874004364, 1.0090399265289307, 1.0088659048080444, 1.0087419629096985, 1.0086515426635743, 1.0085835099220275, 1.0085307478904724, 1.0084882616996764, 1.0084527730941772, 1.008422327041626, 1.008395516872406, 1.0083711862564086, 1.008348536491394, 1.0083271980285644, 1.008307158946991, 1.0082876086235046, 1.0082689285278321, 1.0082506656646728, 1.008232843875885, 1.0082155585289, 1.008198606967926, 1.0081819653511048, 1.0081659317016602, 1.008150041103363, 1.0081344485282897, 1.008119237422943, 1.0081043124198914, 1.0080897331237793, 1.008075201511383, 1.0080609679222108, 1.0080470204353333, 1.008033287525177, 1.0080199003219605, 1.0080066442489624, 1.0079937338829041, 1.0079811334609985, 1.007968580722809, 1.0079561233520509, 1.0079442143440247, 1.0079323530197144, 1.0079205513000489, 1.007909083366394, 1.0078978419303894, 1.0078867316246032, 1.007875645160675, 1.007864785194397, 1.0078542470932006, 1.0078438997268677, 1.0078336596488953, 1.0078237891197204, 1.0078136324882507, 1.0078039765357971, 1.0077943444252013, 1.0077848434448242, 1.0077758193016053, 1.0077664494514464, 1.0077576160430908, 1.0077483177185058, 1.0077397704124451, 1.0077308416366577, 1.0077226400375365, 1.0077142715454102, 1.0077061533927918, 1.0076980113983154, 1.007690179347992, 1.0076820731163025, 1.0076743125915528, 1.0076665997505188, 1.0076591849327088, 1.0076518774032592, 1.0076446652412414, 1.0076375603675842, 1.0076304793357849, 1.0076234817504883, 1.0076165556907655, 1.0076099276542663, 1.0076029539108275, 1.0075965642929077, 1.0075901746749878, 1.0075836658477784, 1.0075776696205139, 1.0075711846351623, 1.007565152645111, 1.0075591206550598, 1.007552981376648, 1.0075472831726073, 1.0075410842895507, 1.0075356364250183, 1.0075299263000488, 1.0075242638587951, 1.0075190544128418, 1.007513415813446, 1.0075080633163451, 1.0075028657913208, 1.0074977040290833, 1.0074954628944397, 1.0074906706809998, 1.0074836611747742, 1.0074782133102418, 1.0074732661247254, 1.0074685096740723, 1.0074635744094849]

x = list(range(0,201,25))
x_ticks = [item * 10 for item in x]

plt.figure(dpi = 600, figsize = (12, 5))
plt.subplot(1, 2, 1)
p1 = plt.plot(M_3_lamda0001, linestyle = '-')
p2 = plt.plot(M_5_lamda0001, linestyle = '-',color = 'orange')
p3 = plt.plot(M_10_lamda0001, linestyle = '-')

plt.xticks(x,x_ticks)
plt.xlabel('steps')
plt.ylabel(r'objective value $\mathcal{L}_{DM}^{\theta} (\Lambda, \mathbb{G})$')
plt.legend([p1,p2,p3], labels =  [r'$M=3,\lambda_2=0.0001,Acc=0.963$',r'$M=5,\lambda_2=0.0001,Acc=0.968$', r'$M=10,\lambda_2=0.0001,Acc=0.956$'], loc = 'best')
plt.title('a')


plt.subplot(1, 2, 2)
p2 = plt.plot(M_5_lamda0001, linestyle = '-',color = 'orange')
M_5_lamda001 = [8.2810640335083, 8.269350051879883, 8.247291851043702, 8.223724842071533, 8.19802122116089, 8.169468688964844, 8.137255096435547, 8.100449466705323, 8.057989120483398, 8.008621644973754, 7.950878095626831, 7.883007144927978, 7.802936601638794, 7.708213520050049, 7.5958771228790285, 7.462275838851928, 7.3030249118804935, 7.113088226318359, 6.887001800537109, 6.619368171691894, 6.305613899230957, 5.943051385879516, 5.532201051712036, 5.07812614440918, 4.591511964797974, 4.088843369483948, 3.5917872905731203, 3.1244424104690554, 2.7045074701309204, 2.3447197914123534, 2.0504467368125914, 1.8196966052055359, 1.645150935649872, 1.5168679714202882, 1.4245910406112672, 1.359189510345459, 1.3132473468780517, 1.2811060428619385, 1.258621299266815, 1.2428400039672851, 1.2316948294639587, 1.22375385761261, 1.2180322051048278, 1.213850498199463, 1.2107431292533875, 1.2083913087844849, 1.206573486328125, 1.2051350712776183, 1.2039705157279967, 1.2030039429664612, 1.20218186378479, 1.2014648914337158, 1.200825309753418, 1.2002419829368591, 1.1996971845626831, 1.1991776823997498, 1.198672890663147, 1.1981721997261048, 1.1976672172546388, 1.1971481442451477, 1.196607530117035, 1.196038019657135, 1.1954339385032653, 1.1947721362113952, 1.1940631985664367, 1.193299376964569, 1.1924806714057923, 1.1916168570518493, 1.1907270908355714, 1.1898401021957397, 1.1889870882034301, 1.1881937026977538, 1.1874644994735717, 1.186780619621277, 1.1861076831817627, 1.185416316986084, 1.1846840143203736, 1.1838935732841491, 1.1829928755760193, 1.1819471716880798, 1.180678415298462, 1.179060971736908, 1.1768791317939757, 1.1737393856048584, 1.1690157532691956, 1.1619417190551757, 1.1519107937812805, 1.139008367061615, 1.1243324398994445, 1.1091750144958497, 1.093955659866333, 1.078930413722992, 1.0647061586380004, 1.0515915393829345, 1.0396467208862306, 1.0288507461547851, 1.0191441297531127, 1.0104565143585205, 1.0027021408081054, 0.9957908034324646, 0.989635682106018, 0.9841561436653137, 0.9792774200439454, 0.9749316155910492, 0.9710577666759491, 0.9676036179065705, 0.9645178079605102, 0.9617597579956054, 0.9592918694019318, 0.95708047747612, 0.9550966799259186, 0.9533145248889923, 0.9517115235328675, 0.9502680599689484, 0.9489664971828461, 0.9477912187576294, 0.9467295467853546, 0.945767617225647, 0.9448961555957794, 0.944105452299118, 0.9433871328830719, 0.9427335977554321, 0.9421383261680603, 0.9415963351726532, 0.9411002337932587, 0.9406473398208618, 0.9402326881885529, 0.9398529946804046, 0.9395044147968292, 0.9391842603683471, 0.9388899087905884, 0.9386189222335816, 0.9383690476417541, 0.938139009475708, 0.9379258692264557, 0.9377286851406097, 0.9375462830066681, 0.9373770892620087, 0.9372201561927795, 0.9370743036270142, 0.9369387149810791, 0.936812448501587, 0.9366946041584014, 0.9365843653678894, 0.9364814102649689, 0.9363851904869079, 0.9362950026988983, 0.9362103819847107, 0.9361309468746185, 0.936056125164032, 0.9359858989715576, 0.9359200298786163, 0.9358571350574494, 0.9357979834079743, 0.9357426524162292, 0.9356893718242645, 0.9356388092041016, 0.9355910778045654, 0.935545700788498, 0.935502576828003, 0.9354613125324249, 0.9354220807552338, 0.9353847026824951, 0.9353488504886627, 0.9353145062923431, 0.9352816045284271, 0.935250872373581, 0.9352202653884888, 0.9351907134056091, 0.9351625442504883, 0.9351353406906128, 0.935109144449234, 0.9350839078426361, 0.9350594699382782, 0.9350358843803406, 0.9350130617618561, 0.9349913656711578, 0.9349697291851043, 0.9349483907222748, 0.93492791056633, 0.934908103942871, 0.9348888516426086, 0.9348700761795044, 0.9348517715930938, 0.9348339200019836, 0.934816426038742, 0.9347995042800903, 0.9347828507423401, 0.9347664952278137, 0.9347505986690521, 0.9347347915172577, 0.9347195029258728, 0.9347043931484222, 0.9346905171871185, 0.934681522846222, 0.9346628248691559, 0.934648197889328, 0.9346340239048004, 0.934620600938797, 0.9346075415611267, 0.9345946729183197]

M_5_lamda00001 = [8.2810640335083, 8.269300270080567, 8.246946907043457, 8.22281255722046, 8.196288013458252, 8.166665172576904, 8.133125019073486, 8.094712162017823, 8.05029125213623, 7.998498439788818, 7.937669277191162, 7.8657914161682125, 7.780426931381226, 7.678657960891724, 7.556997537612915, 7.41135458946228, 7.237078523635864, 7.029126787185669, 6.782124137878418, 6.490994262695312, 6.151767063140869, 5.762705755233765, 5.32684736251831, 4.850541353225708, 4.345136213302612, 3.828061509132385, 3.320697617530823, 2.8450133085250853, 2.419718360900879, 2.0570006132125855, 1.760895574092865, 1.5284142732620238, 1.3517719030380249, 1.2209769129753112, 1.1259470939636231, 1.0577955007553101, 1.00930095911026, 0.9749270856380463, 0.950580894947052, 0.9333130240440368, 0.9210293054580688, 0.9122543275356293, 0.9059513211250305, 0.9013972103595733, 0.8980813682079315, 0.8956472158432007, 0.8938424825668335, 0.8924947261810303, 0.8914673388004303, 0.8906772494316101, 0.8900607526302338, 0.8895714104175567, 0.8891754865646362, 0.8888472378253937, 0.888570362329483, 0.8883315801620484, 0.88812415599823, 0.887932825088501, 0.8877554118633271, 0.88758425116539, 0.8874215304851532, 0.8872629523277282, 0.8871059477329254, 0.8869469344615937, 0.8867858946323395, 0.886627596616745, 0.8864545166492462, 0.8862695574760437, 0.8860808253288269, 0.8858813047409058, 0.8856697916984558, 0.8854444265365601, 0.8852023482322693, 0.8849419772624969, 0.8846642851829529, 0.8843664824962616, 0.8840230762958526, 0.8836532890796661, 0.8832462668418884, 0.8827944576740265, 0.8822891175746918, 0.8817188382148743, 0.8810712397098541, 0.8803277671337127, 0.8794672608375549, 0.8784715950489044, 0.8772775888442993, 0.8758381009101868, 0.874091774225235, 0.8719307780265808, 0.8692054152488708, 0.865699052810669, 0.861088651418686, 0.854885584115982, 0.8463543593883515, 0.8343977332115173, 0.8174254536628723, 0.7933340311050415, 0.7598201990127563, 0.7156644821166992, 0.6630457460880279, 0.6082614719867706, 0.5571848332881928, 0.510633236169815, 0.4673245996236801, 0.4279361516237259, 0.39301859140396117, 0.36222758889198303, 0.3350922465324402, 0.31114000976085665, 0.2899347573518753, 0.27110545337200165, 0.2543300300836563, 0.23932938277721405, 0.22586511373519896, 0.21373563408851623, 0.20276983231306075, 0.19282203167676926, 0.18376805484294892, 0.1755015715956688, 0.16793130785226823, 0.16097905337810517, 0.15457711219787598, 0.14866657555103302, 0.14319660067558287, 0.13812242895364762, 0.13340510427951813, 0.12901046872138977, 0.12490804195404052, 0.12107134088873864, 0.1174764797091484, 0.1141025334596634, 0.11093081012368203, 0.1079445593059063, 0.10512860119342804, 0.10246945768594742, 0.09995502680540085, 0.09757432267069817, 0.0953171856701374, 0.09317491427063943, 0.09113915637135506, 0.08920245096087456, 0.08735815361142159, 0.08559995740652085, 0.08392214253544808, 0.08231965675950051, 0.08078775480389595, 0.07932181656360626, 0.07791793197393418, 0.07657236829400063, 0.0752817563712597, 0.07404277846217155, 0.07285247817635536, 0.07170812264084817, 0.0706072673201561, 0.06954745203256607, 0.06852649822831154, 0.06754247471690178, 0.066593337059021, 0.06567736715078354, 0.06479282453656196, 0.0639382354915142, 0.06311211809515953, 0.06231309399008751, 0.06153996102511883, 0.060791248455643654, 0.06006601192057133, 0.05936315581202507, 0.05868169069290161, 0.058020664379000664, 0.05737913362681866, 0.05675626993179321, 0.0561513215303421, 0.055563480406999585, 0.05499204434454441, 0.054436386376619336, 0.05389569252729416, 0.0533694926649332, 0.05285729765892029, 0.05235852003097534, 0.05187257565557957, 0.051399017497897145, 0.050937331095337865, 0.05048709399998188, 0.05004792660474777, 0.04961936958134174, 0.0492009986191988, 0.048792599141597746, 0.04839363507926464, 0.04800389744341373, 0.04762301407754421, 0.047250650450587274, 0.04688658826053142, 0.046530558913946155, 0.04618219807744026, 0.04584132395684719, 0.045507689192891124, 0.045181038230657576, 0.04486117921769619, 0.044547881186008456, 0.04424097388982773]

p1 = plt.plot(M_5_lamda001, linestyle = '-',color = 'purple')
p3 = plt.plot(M_5_lamda00001,color = 'lightseagreen')
plt.xticks(x,x_ticks)
plt.xlabel('steps')
plt.ylabel(r'objective value $\mathcal{L}_{DM}^{\theta} (\Lambda, \mathbb{G})$')
plt.legend([p1,p2,p3], labels =  [r'$M=5,\lambda_2=0.001,Acc=0.953$',r'$M=5,\lambda_2=0.0001,Acc=0.968$', r'$M=5,\lambda_2=0.00001,Acc=0.962$'], loc = 'best')
plt.show()
plt.title('b')
plt.savefig('ab_DM.eps', dpi = 600, format = 'eps')
