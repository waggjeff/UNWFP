#!/usr/bin/env python
# coding: utf-8

# ## Market Functionality Index Analysis
# 
# We want to reproduce the Market Functionality Index (MFI) analysis, including spider charts. 
# 
# 
# ### Import required packages 
# 
# Run the first code block (SHIFT-RETURN) to import the required Python packages. 

# In[13]:


# import the required packages 
import plotly.express as px
import pandas as pd
from numpy import nan
from ipywidgets import interact, interactive
import PySimpleGUI as sg
from statistics import mean
import math
import warnings
warnings.filterwarnings('ignore')


# ### Read and format survey data
# 
# To begin, ensure that the completed 'MFI_Full_vX' spreadsheet is in the working directory and enter the filename HERE:
df = pd.read_csv("MFI_Full_v1_10_Aug2023.csv")

# check the names of all of the markets that have been surveyed, and create an array of unique values
Markets = df.MarketName.unique()

# change the column names of important model queries 
# Note that - there are multiple E1 and E2 variables in the xlsx spreadsheet (CHECK)
#           - There is no 'MktSkuNb_Cl' column in the DF
df.rename(columns = {'UOASoldGroup_Gr':'A1','TrdSkuNb_Cl':'A2_T','MktSkuNb_Cl':'A2_M','UOAAvailScarce_Gr':'B1',
                    'TrdAvailRunout_Gr':'B2_T','MktAvailRunout_Gr':'B2_M','UOAPriceIncr_Gr':'C1',
                    'TrdPriceStab_Gr':'C2_T','MktPriceStab_Gr':'C2_M','TrdResilStockout':'D1p1',
                    'TrdResilLeadtime':'D1p2','TrdResilNodDens_Gr':'D2p1','TrdResilNodComplex_Gr':'D2p2',
                    'TrdResilNodCrit_Gr':'D2p3','TrdServiceShopExp':'E1','TrdServiceCheckoutExp':'E2',
                    'TrdStructureType':'F1_T','MktStructureType':'F1_M','TrdStructureCond':'F2_T',
                    'MktStructureCond':'F2_M','UOAStructureFeat':'F3',
                    'MktCompetLessFive_Gr':'G1','MktCompetOneContr_Gr':'G2',
                    'MktTraderNb':'G3','UOAQltyFood':'H1','UOAQltyFVegFruSeparate':'H2','UOAQltyFAnimRefrig':'H3p1',
                    'UOAQltyFAnimRefrigWork':'H3p2','UOAQltyPackExpiry':'H4','UOAQltyFPackGood':'H5p1',
                    'UOAQltyFVegFruGood':'H5p2','UOAQltyPlastGood':'H5p3','MktAccessCnstr':'I1',
                    'MktProtCnstr':'I2'}, inplace = True)

# ### Choose the region and calculate the scores for each dimension 
# 
# The user must first decide which market region they would like to analyse. Run the next two blocks of code. 

# In[16]:

# Create a pop-up window and select region
pfont = ("Arial", 18)
layout = [ 
            [sg.Text('Select region: ',font = ("Arial", 18)),
             sg.Listbox(sorted(Markets),select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,size=(20,len(Markets)),font = pfont)],
            [sg.Button('Ok',font = pfont), sg.Button('Cancel',font = pfont)]
        ]

# Create the Window
window = sg.Window('Select the region', layout)

# Event Loop to process "events" and get the "values" of the input
while True:
    event, values = window.read()
    print( f"event={event}" )
    if event is None or event == 'Ok' or event == 'Cancel': # if user closes window or clicks cancel
        break
        
# close  the window        
window.close()

if event == "Cancel":
    print( "You cancelled" )
    exit()
else:
    print('You entered ', values[0][0])
    sg.popup( f"You selected {values[0][0]}",font=pfont)

# After selecting the region of interest from the pulldown menu, run the following code block to calculate the dimensional scores and plot the spider plot. 

# Retrieve the desired market and fiter the results to create a new dataframe
mfi_market =  values[0][0]#w.result
df_mfi = df.loc[df['MarketName'] == mfi_market]

# calculate the score for Assortment - A
a1sum = []
for a in df_mfi.A1:
    atmp = []
    if a == '9999' or a == 9999.0:
        atmp = [0]
        a = float(a)
    if isinstance(a, str):
        atmp = a.split()
        
    aint = [int(i) for i in atmp]
    aint = [item for item in aint if not(math.isnan(item)) == True]
    if len(aint) > 0:
        a1sum.append(sum(aint))

# number of distinct objects sold. Include? 
a2sum = [x for x in df_mfi.A2_T if not math.isnan(x)]

# (26/09/2023 - removed A2 variable for consistency with HQ version)
A = mean(a1sum) + mean(a2sum) #+ mean(a2sum)
#A = (mean(a1sum) + mean(a2sum)) * 10. / 6.

# calculate the score for Availability - B (Note that polarity is negative so a high score 
# indicates market is not functioning well )
b1sum = []
for b in df_mfi.B1:
    btmp = []
    if b == '9999' or b == 9999.0:
        btmp = [0]
        b = float(b)
    if isinstance(b, str):
        btmp = b.split()
    bint = [int(i) for i in btmp]
    bint = [item for item in bint if not(math.isnan(item)) == True]
    if len(bint) > 0:
        b1sum.append(sum(bint))
    
b2sum = []
for b in df_mfi.B2_M:
    btmp = []
    if b == '9999' or b == 9999.0:
        btmp = [0]
        b = float(b)
    if isinstance(b, str):
        btmp = b.split()
    bint = [int(i) for i in btmp]
    bint = [item for item in bint if not(math.isnan(item)) == True]
    if len(bint) > 0:
        b2sum.append(sum(bint))

b3sum = []
for b in df_mfi.B2_T:
    btmp = []
    if b == '9999' or b == 9999.0:
        btmp = [0]
        b = float(b)
    if isinstance(b, str):
        btmp = b.split()
    bint = [int(i) for i in btmp]
    bint = [item for item in bint if not(math.isnan(item)) == True]
    if len(bint) > 0:
        b3sum.append(sum(bint))

# removed trader data for consitency with HQ calculations 
B = 0 
if (len(b1sum) > 0 and len(b2sum) > 0 and len(b3sum) >0):
    B = ((6-mean(b1sum)) + (6-mean(b2sum)) +  (6-mean(b3sum)))* 10./18.
if (len(b1sum) > 0 and len(b2sum) == 0 and len(b3sum) > 0):
    B = ((6-mean(b1sum)) + (6-mean(b3sum)))* 10./12.
    print("No B2 Market data.")
if (len(b1sum) == 0 and len(b2sum) > 0 and len(b3sum) > 0):
    B = ((6-mean(b2sum)) + (6-mean(b3sum))) * 10./12.
    print("No B1 data.")
if (len(b1sum) > 0 and len(b2sum) > 0 and len(b3sum) == 0):
    B = ((6-mean(b1sum)) + (6-mean(b2sum)))* 10./12.
    print("No B2 Trader data.")
#B = ((6-mean(b1sum)) + (6-mean(b2sum)))* 10./12.
    
# calculate the score for Prices - C
c1sum = []
for c in df_mfi.C1:
    ctmp = []
    if c == '9999' or c == 9999.0:
        ctmp = [0]
        c = float(c)
    if isinstance(c, str):
        ctmp = c.split()
    
    cint = [int(i) for i in ctmp]
    cint = [item for item in cint if not(math.isnan(item)) == True]
    if len(cint) > 0: 
        c1sum.append(sum(cint)) # change to len()

c2sum = []
for c in df_mfi.C2_M:
    ctmp = []
    if c == '9999' or c == 9999.0:
        ctmp = [0]
        c = float(c)
    if isinstance(c, str):
        ctmp = c.split()
    
    cint = [int(i) for i in ctmp]
    cint = [item for item in cint if not(math.isnan(item)) == True]
    if len(cint) > 0:
        c2sum.append(sum(cint)) # change to len()

c3sum = []
for c in df_mfi.C2_T:
    ctmp = []
    if c == '9999' or c == 9999.0:
        ctmp = [0]
        c = float(c)
    if isinstance(c, str):
        ctmp = c.split()
    
    cint = [int(i) for i in ctmp]
    cint = [item for item in cint if not(math.isnan(item)) == True]
    if len(cint) > 0:
        c3sum.append(sum(cint)) # change to len()
        
# polarity of C1 is negative, while C2_T and C2_M are positive
C = 0
if (len(c1sum) > 0 and len(c2sum) > 0 and len(c3sum) > 0):
    C = ((6. - mean(c1sum)) + mean(c2sum)  + mean(c3sum))* 10./18.
if (len(c1sum) > 0 and len(c2sum) == 0 and len(c3sum) > 0):
    C = ((6. - mean(c1sum)) + mean(c3sum))* 10./12.
    print("No C2 Market data.")
if (len(c1sum) == 0 and len(c2sum) > 0 and len(c3sum) > 0):
    C = (mean(c2sum) + mean(c3sum))* 10./12.
    print("No C1 data.")
if (len(c1sum) > 0 and len(c2sum) > 0 and len(c3sum) == 0):
    C = ((6. - mean(c1sum)) + mean(c2sum))* 10./12.
    print("No C2 Trader data.")
    
# calculate the score for Resilience - D  (Note: need to incorporate D2.1 and D2.3)
d1sum = df_mfi.D1p1 + df_mfi.D1p2
d1sum = [x for x in d1sum if not math.isnan(x)]

d2p1sum = []
for d in df_mfi.D2p1:
    dtmp = []
    if d == '9999' or d == 9999.0:
        dtmp = [0]
        d = float(d)
    if isinstance(d, str):
        dtmp = d.split()
    
    dint = [int(i) for i in dtmp]
    dint = [item for item in dint if not(math.isnan(item)) == True]
    d2p1sum.append(6. - sum(dint)) # due to negative polarity with maximum 6 

d2p2sum = []
for d in df_mfi.D2p2:
    dtmp = []
    if d == '9999' or d == 9999.0:
        dtmp = [0]
        d = float(d)
    if isinstance(d, str):
        dtmp = d.split()
    
    dint = [int(i) for i in dtmp]
    dint = [item for item in dint if not(math.isnan(item)) == True]
    d2p2sum.append(sum(dint)) # positive polarity, higher number is good, max 6

d2p3sum = []
for d in df_mfi.D2p3:
    dtmp = []
    if d == '9999' or d == 9999.0:
        dtmp = [0]
        d = float(d)
    if isinstance(d, str):
        dtmp = d.split()
    
    dint = [int(i) for i in dtmp]
    dint = [item for item in dint if not(math.isnan(item)) == True]
    d2p3sum.append(6. - sum(dint)) # due to negative polarity with maximum 6

d2avg = (mean(d2p1sum) + mean(d2p2sum) + mean(d2p3sum))/ 3.
    
D = 0
D = 0.5*((mean(d1sum)) + d2avg) * 10./4.

# calculate the score for Service - E (formerly competition). Assume that counts of yes answers indicates 
#             positive polarity with yes/no answers (Need to verify that these are the right E1/E2 queries
#             as they are different in the tech guidance)
e1sum = []
for e in df_mfi.E1:
    etmp = []
    if e == '9999' or e == 9999.0:
        etmp = [0]
        e = float(e)
    if isinstance(e, str):
        etmp = e.split()
    
    eint = [int(i) for i in etmp]
    eint = [item for item in eint if not(math.isnan(item)) == True]
    if len(eint) > 0:
        e1sum.append(len(eint))

e2sum = []
for e in df_mfi.E2:
    etmp = []
    if e == '9999' or e == 9999.0:
        etmp = [0]
        e = float(e)
    if isinstance(e, str):
        etmp = e.split()
    
    eint = [int(i) for i in etmp]
    eint = [item for item in eint if not(math.isnan(item)) == True]
    if len(eint) > 0:
        e2sum.append(len(eint))
        
E = 0
if (len(e1sum) > 0 and len(e2sum) > 0):
    E = 0.5*(mean(e1sum) + mean(e2sum))* 10./3.
if (len(e1sum) > 0 and len(e2sum) == 0):
    E = mean(e1sum)* 10./3.
    print("No E2 data.")
if (len(e1sum) == 0 and len(e2sum) > 0):
    E = mean(e2sum)* 10./3.
    print("No E1 data.")

# calculate the score for Infrastructure - F  (need to decided whether to use both Market and Trader data)
f1sum = []
for f in df_mfi.F1_M:
    ftmp = []
    if f == '9999' or f == 9999.0:
        continue
    if isinstance(f, str):
        ftmp = f.split()
    else:
        continue
    fint = [int(i) for i in ftmp]
    fint = [item for item in fint if not(math.isnan(item)) == True]
    if len(fint) > 0:
        f1sum.append(mean(fint))
        
f2sum = []
for f in df_mfi.F2_M:
    ftmp = []
    if f == '9999' or f == 9999.0:
        continue
    if isinstance(f, str):
        ftmp = f.split()
    else:
        continue
    fint = [int(i) for i in ftmp]
    fint = [item for item in fint if not(math.isnan(item)) == True]
    if len(fint) > 0:
        f2sum.append(mean(fint))

f3sum = []
for f in df_mfi.F3:
    ftmp = []
    if f == '9999' or f == 9999.0:
        continue
    if isinstance(f, str):
        ftmp = f.split()
    else:
        continue
    fint = [int(i) for i in ftmp]
    fint = [item for item in fint if not(math.isnan(item)) == True]
    if len(fint) > 0:
        f3sum.append(len(fint))
        
try:
    f1avg = 10.*mean(f1sum) / 3.
    f2avg = 10.*mean(f2sum) / 3.
    f3avg = 10.*mean(f3sum) / 8.

    F = (f1avg + f2avg + f3avg)/ 3.
except:
    F = 0
    
# calculate the score for Competition - G (formerly Service)
g1sum = []
for g in df_mfi.G1:
    gtmp = []
    if g == '9999' or g == 9999.0:
        gtmp = [0]
        g = float(g)
    if isinstance(g, str):
        gtmp = g.split()
    if isinstance(g, float) and g >= 1.0 and g <= 3.0:
        gtmp = [g]
        
    gint = [int(i) for i in gtmp]
    gint = [item for item in gint if not(math.isnan(item)) == True]
    if len(gint) > 0:
        g1sum.append(sum(gint))

g2sum = []
for g in df_mfi.G2:
    gtmp = []
    if g == '9999' or g == 9999.0:
        gtmp = [0]
        g = float(g)
    if isinstance(g, str):
        gtmp = g.split()
    if isinstance(g, float) and g >= 1.0 and g <= 3.0:
        gtmp = [g]
        
    gint = [int(i) for i in gtmp]
    gint = [item for item in gint if not(math.isnan(item)) == True]
    if len(gint) > 0:
        g2sum.append(sum(gint))
    
for g in df_mfi.index:
    gtmp = []
    if isinstance(df_mfi.G3[g], str):
        gtmp = df_mfi.G3[g].split()
        gtmp = [float(x) for x in gtmp]
        df_mfi.at[g, 'G3'] = mean(gtmp)

g3 = [item for item in df_mfi.G3.astype(float) if not(math.isnan(item)) == True]
g3 = [5 if item >= 5. else item for item in g3]

try:
    G = 10.*((6. - mean(g1sum))/6. + (6. - mean(g2sum))/6. + mean(g3)/5.) / 3.
except:
    G = 0
      
# calculate the score for Food Quality - H    
h1 = [item for item in df_mfi.H1 if (not(math.isnan(item)) == True and item != 9999.)]
if len(h1) > 0:
    h1sum = sum(h1) / len(h1)
else:
    h1sum = float("NaN")
h2 = [item for item in df_mfi.H2 if (not(math.isnan(item)) == True and item != 9999.)]
if len(h2) > 0:
    h2sum = sum(h2) / len(h2)
else:
    h2sum = float("NaN")
h3p1 = [item for item in df_mfi.H3p1 if (not(math.isnan(item)) == True and item != 9999.)]
if len(h3p1) > 0:
    h3p1sum = sum(h3p1) / len(h3p1)
else:
    h3p1sum = float("NaN")
h3p2 = [item for item in df_mfi.H3p2 if (not(math.isnan(item)) == True and item != 9999.)]
if len(h3p2) > 0:
    h3p2sum = sum(h3p2) / len(h3p2)
else:
    h3p2sum = float("NaN")
h4 = [item for item in df_mfi.H4 if (not(math.isnan(item)) == True and item != 9999.)]
if len(h4) > 0:
    h4sum = sum(h4) / len(h4)
else:
    h4sum = float("NaN")
h5p1 = [item for item in df_mfi.H5p1 if (not(math.isnan(item)) == True and item != 9999.)]
if len(h5p1) > 0:
    h5p1sum = sum(h5p1) / len(h5p1)
else:
     h5p1sum = float("NaN")
h5p2 = [item for item in df_mfi.H5p2 if (not(math.isnan(item)) == True and item != 9999.)]
if len(h5p2):
    h5p2sum = sum(h5p2) / len(h5p2)
else:
    h5p2sum = float("NaN")
h5p3 = [item for item in df_mfi.H5p3 if (not(math.isnan(item)) == True and item != 9999.)]
if len(h5p3) > 0:
    h5p3sum = sum(h5p3) / len(h5p3)
else:
    h5p3sum = float("NaN")
    
hvals = [h1sum, h2sum, h3p1sum, h3p2sum, h4sum, h5p1sum, h5p2sum, h5p3sum]
hclean =  [x for x in hvals if str(x) != 'nan']
H = 10.*mean(hclean)


# calculate the score for Access & Protection - I
for i in df_mfi.index:
    
    if isinstance(df_mfi.I1[i], str):
        itmp = df_mfi.I1[i].split()
        itmp = [float(x) for x in itmp]
        df_mfi.at[i, 'I1'] = mean(itmp)
    if isinstance(df_mfi.I2[i], str):
        itmp = df_mfi.I2[i].split()
        itmp = [float(x) for x in itmp]
        df_mfi.at[i, 'I2'] = mean(itmp)
        
i1 = [item for item in df_mfi.I1.astype(float) if not(math.isnan(item)) == True]
i1 = [0 if item == 9999. else item for item in i1]
i2 = [item for item in df_mfi.I2.astype(float) if not(math.isnan(item)) == True]
i2 = [0 if item == 9999. else item for item in i2]

try: 
    i1revpol = 10.*(3. - mean(i1))/3.
    i2revpol = 10.*(3. - mean(i2))/3.

    I = (i1revpol + i2revpol)/2.
except: 
    I = 0
    
# Now calculate the MFI from the dimension scores
mfi_dims = [A, B, C, D, E, F, G, H, I]
alpha = 0.5
beta = 1.
mu = mean(mfi_dims)

MFI = mu - alpha*(math.sqrt((mu - min(mfi_dims)) + beta**2.) - beta)

# Sample data
plotdf = pd.DataFrame(dict(
    value = [A, B, C, D, G, F, E, H, I],
    variable = ['Assortment', 'Availability', 'Price', 'Resilience', 'Competition',
                'Infrastructure','Service','Food Quality','Access & Protection']))       

# Plot the data 
fig = px.line_polar(plotdf, r = 'value', theta = 'variable', line_close = True, 
                    title="Region: "+mfi_market+" ( MFI = {mfi_t:.1f} )".format(mfi_t=MFI))

fig.show()


# In[ ]:




