#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Titulo:625 fechas que coinciden entre WRF y GFS  
#Autor: LUISA FERNANDA BURITICÁ RUÍZ
#Febrero 2023
#fernanda.buritica@udea.edu.co
#PROYECTO 7 SAMA

# 1. Comparación entre GFS y GPM en la resolución de GFS. 
# 2. Comparación entre WFR y GPM en la resolución de GPM.
# 3. Comparar estaciones en tierra con Sonsón y Rionegro en GPM, GFS y WRF
###############################################################################
#1. LIBRERIAS
#https://pankajkarman.github.io/bias_correction/index.html#usage
from bias_correction import BiasCorrection, XBiasCorrection
import glob
import matplotlib.pyplot as plt  # Generación de gráficos
import matplotlib
import matplotlib as mpl
from matplotlib import cm
import numpy             as np   # Herramienta matemática.
import os                        # Leer archivos dentro de una carpeta.
import pandas            as pd   # Procesar datos. 
from   tqdm    import tqdm       # Tiempo de ejecución.
###############################################################################
# 2. FUNCIONES
# 2.1 fechas
def P1FECHAS(df,ncf):#Proceso general e inicial para convertir los datos a Datetime.
    df[ncf]=pd.to_datetime(df[ncf])
    df["year"]=pd.to_datetime(df[ncf]).dt.year   # crea una columna con los años.
    df["month"]=pd.to_datetime(df[ncf]).dt.month # crea una columna con los meses.
    df["day"]=pd.to_datetime(df[ncf]).dt.day     # crea una columna con los dias.
    df["hour"]=pd.to_datetime(df[ncf]).dt.hour   # crea una columna con los hora.
    return(df)
# 2.2 procesamiento de datos de entrada
#df1,nameadicion=GPM,"GPM"
def IngesarSeriesTiempo(df1,nameadicion):
    df2=pd.DataFrame()
    for i in range(len(df1)):
        df=pd.read_csv(df1[i],names=["valor"])
        municipio=df1[i].split("/")[9].split("_")[5]+nameadicion
        print(municipio)
        df.columns=[municipio]
        df2=pd.concat([df2,df[municipio]],axis=1)
    return(df2)
# 2.3 Tablas de contingencia
def rellenar_evento(variable,ncv,evento_precipitacion,percentil):#Clasifica la lluvia y no lluvia
    variable[evento_precipitacion]=np.nan
    variable[evento_precipitacion][variable[ncv]>percentil]=1
    variable[evento_precipitacion][variable[ncv]<=percentil]=0
    return(variable)

def TablaContingencia(V,ncv,ncf,V1,ncv1):# Entrega los parametros de la tabla de contingencia
    v=pd.concat([V[ncf],V[ncv],V1[ncv1]],axis=1)
    #EP=Evaluación percentil
    per1=np.percentile(v[ncv][v[ncv]>0],5) #Percentil 5 de los datos mayores a cero Observación
    per2=np.percentile(v[ncv1][v[ncv1]>0],5) #Percentil 5 de los datos mayores a cero Pronóstico
    v= rellenar_evento(v,ncv,"EPobs",per1) 
    v= rellenar_evento(v,ncv1,"EPpro",per2) 
    #v[ncv][v[ncv]<0]=0
    #v[ncv1][v[ncv1]<0]=0
    
    n=v["EPobs"][(v[ncv]>=0) & (v[ncv1]>=0)].count()
    #dfprueba=v["EPobs"][(v[ncv]>=0)].count()
    #dfprueba2=v["EPobs"][(v[ncv1]>=0)].count()
    Errodfncv=v[ncv][(v[ncv]<0)]
    #Errodfncv1=v["EPobs"][(v[ncv1]<0)].count()
    print(Errodfncv)
    print(type(Errodfncv))
    #print(ncv,ncv1)
    #print("\n numero de datos {}- completo {}- >0 {} -{}".format(ncv,len(v[ncv]),dfprueba,dfprueba2))
    #print("\n Valores que no cumplen\n {}\n{}".format(Errodfncv,Errodfncv1))
    A=v["EPobs"][(v["EPobs"]==1) & (v["EPpro"]==1)].count()
    B=v["EPobs"][(v["EPobs"]==0) & (v["EPpro"]==1)].count()
    C=v["EPobs"][(v["EPobs"]==1) & (v["EPpro"]==0)].count()
    D=v["EPobs"][(v["EPobs"]==0) & (v["EPpro"]==0)].count()
    DATOS=[n,A,B,C,D]
    return(DATOS)
def Escalares(n,A,B,C,D): #Obtiene los valores de los escalares.
    PC=round((A+D)/(n),3)
    Bias=round((A+B)/(A+C),3)
    FAR=round((B)/(A+B),3)
    POD=round((A)/(A+C),3)
    farpod=round(FAR/POD,3)
    podfar=round(POD/FAR,3)
    return(PC,Bias,FAR,POD,farpod,podfar)
def union(ve1,ve2):#Une los parametros con los escalares de la tabla de contingencia.
    for i in range(0,len(ve2)):
        ve1.append(ve2[i])
    return(ve1) 
#df,ncv1,ncv2,ncf=df,M[0]+ncv1,M[0]+ncv2,"fecha"
def TCFinal(df,ncv1,ncv2,ncf): #Valores de la tabla de contingencia, entrega un vectore
    print(ncv1," vs ",ncv2)    
    m1=TablaContingencia(df,ncv1,ncf,df,ncv2)#
    m5= Escalares(m1[0],m1[1],m1[2],m1[3],m1[4])
    m1=union(m1,m5)
    return(m1)
#df,ncv1,ncv2=datos1,"GPM","GFS"
def VectoresFinales(df,ncv1,ncv2): #Entrega los tres vectores finales de comparación
    M=["Sonson","Anza","Apartado","Caceres","Cisneros","Jerico","PuertoTriunfo",
       "Salgar","SanPedro"]
    m1=TCFinal(df,M[0]+ncv1,M[0]+ncv2,"fecha")
    m2=TCFinal(df,M[1]+ncv1,M[1]+ncv2,"fecha")
    m3=TCFinal(df,M[2]+ncv1,M[2]+ncv2,"fecha")
    m4=TCFinal(df,M[3]+ncv1,M[3]+ncv2,"fecha")
    m5=TCFinal(df,M[4]+ncv1,M[4]+ncv2,"fecha")
    m6=TCFinal(df,M[5]+ncv1,M[5]+ncv2,"fecha")
    m7=TCFinal(df,M[6]+ncv1,M[6]+ncv2,"fecha")
    m8=TCFinal(df,M[7]+ncv1,M[7]+ncv2,"fecha")
    m9=TCFinal(df,M[8]+ncv1,M[8]+ncv2,"fecha")
    #San Pedro, Sonsón, Jericó, Salgar, Cisneros, Anzá, Puerto T., caceres, Apartadó
    v1=[m9,m1,m6,m8,m5,m2,m7,m4,m3]
    return(v1)
def SeleccionEventosSinNan(df,ncv3,municipio):
    M=["Sonson","Rionegro"]
    df3=[]
    for index, row in (df.iterrows()):
        if (np.isnan(df[M[municipio]][index]) != True and 
            np.isnan(df[M[municipio]+ncv3][index]) != True):
            df3.append([row["fecha"],row[M[municipio]],row[M[municipio]+ncv3]])
    df3=pd.DataFrame(df3,columns=["fecha",M[municipio],M[municipio]+ncv3])
    print(len(df3))
    return(df3)   
#df,ncv1,ncv2=datos6,"GPM","GFS"
#df,ncv1,ncv2=datos7,"GPM","WRF"
def VectoresFinalesOBS(df,ncv1,ncv2): #Entrega los tres vectores finales de comparación
    M=["Sonson","Rionegro"]
    
    dfSNCV1=SeleccionEventosSinNan(df,ncv1,0) #Sonsón Obs vs Sonsón model ncv1
    dfSNCV2=SeleccionEventosSinNan(df,ncv2,0) #Sonsón Obs vs Sonsón model ncv2
    dfRNCV1=SeleccionEventosSinNan(df,ncv1,1) #Rionegro Obs vs Rionegro model ncv1
    dfRNCV2=SeleccionEventosSinNan(df,ncv2,1) #Rionegro Obs vs Rionegro model ncv2
    
    #Sonson
    m1=TCFinal(df,M[0]+ncv1,M[0]+ncv2,"fecha")# directo Modelo ncv1 vs ncv2
    m2=TCFinal(dfSNCV1,M[0],M[0]+ncv1,"fecha") #filtrado obs vs ncv1
    m3=TCFinal(dfSNCV2,M[0],M[0]+ncv2,"fecha") #filtrado obs vs ncv1
    
    #Rionegro
    m4=TCFinal(df,M[1]+ncv1,M[1]+ncv2,"fecha")# directo Modelo ncv1 vs ncv2
    m5=TCFinal(dfRNCV1,M[1],M[1]+ncv1,"fecha") #filtrado obs vs ncv1
    m6=TCFinal(dfRNCV2,M[1],M[1]+ncv2,"fecha") #filtrado obs vs ncv1
    v1=[m1,m2,m3,m4,m5,m6]
    return(v1)
def vectorfinal2(df):
    print("\n",df.isnull().sum(),"\n")
    m1 =TCFinal(df,"obs","model","fecha")
    m2 =TCFinal(df,"obs","gamma","fecha")
    m5=[m1,m2]
    return(m5)
#2.4 Graficos y tablas
def vector5(df,maxi,posicion):
    df1=[]
    for i in range(0,maxi):
        df1.append(df[i][posicion])
    df1=pd.DataFrame(df1)
    return(df1)    
def g1(df,n1,n2,n3,fig,t1,t2,cont): #Modelos
    #Información de entrada 
    filas=["San Pedro", "Sonsón", "Jericó", "Salgar", "Cisneros", "Anzá", 
           "Puerto T.", "caceres", "Apartadó"]
    columnas = ["n","A","B","C","D","PC","Bias","FAR","POD","FAR/POD","POD/FAR"]    
    #Creación de vectores
    #PC, FAR y POD
    vPC=vector5(df,len(filas),5)
    vFAR=vector5(df,len(filas),7)
    vPOD=vector5(df,len(filas),8)
    #Gráfico
    #Ubicación del Subplot "Sub grafico"
    ax = fig.add_subplot(n1,n2,n3)
    #Titulos y labels de ejes
    ax.set_title(t1+"\n",fontsize=15)
    #Gráfico
    tabla = ax.table(df,colWidths=[0.05] * len(columnas),rowLabels=filas, colLabels=columnas,
                     rowColours=["plum"]*len(filas),colColours=["plum"]*len(columnas),
                     cellLoc ='center', loc ="center")
    #Configuraciones del tamaño del gráfico
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(2, 2)
    # Removing ticks and spines enables you to get the figure only with table
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    #configuración de formato condicional por columna
    def formatocondicional(vectorcolumna,ubicacion,cond2):
        vals =vectorcolumna
        normal = cm.colors.Normalize(vals.min()[0], vals.max()[0])
        if cond2==1:
            bcmap2 = plt.cm.spring(normal(vals))
        if cond2==2:
            bcmap2 = plt.cm.Spectral(normal(vals))
        for idx, bb in enumerate(bcmap2):
            tabla[(idx+1, ubicacion)].set_facecolor(bb) 
    formatocondicional(vPC,5,1)
    formatocondicional(vFAR,7,1)
    formatocondicional(vPOD,8,1) 
def g2(df,n1,n2,n3,fig,t1,t2,cont,ncv1,ncv2): #Modelos
    #Información de entrada 
    filas=["S-"+ncv1+"vs"+ncv2,"S-OBSvs"+ncv1,"S-OBSvs"+ncv2,
           "R-"+ncv1+"vs"+ncv2,"R-OBSvs"+ncv1,"R-OBSvs"+ncv2]
    columnas = ["n","A","B","C","D","PC","Bias","FAR","POD","FAR/POD","POD/FAR"]    
    #Creación de vectores
    #PC, FAR y POD
    vPC=vector5(df,len(filas),5)
    vFAR=vector5(df,len(filas),7)
    vPOD=vector5(df,len(filas),8)
    #Gráfico
    #Ubicación del Subplot "Sub grafico"
    ax = fig.add_subplot(n1,n2,n3)
    #Titulos y labels de ejes
    ax.set_title(t1+"\n",fontsize=15)
    #Gráfico
    tabla = ax.table(df,colWidths=[0.05] * len(columnas),rowLabels=filas, colLabels=columnas,
                     rowColours=["plum"]*len(filas),colColours=["plum"]*len(columnas),
                     cellLoc ='center', loc ="center")
    #Configuraciones del tamaño del gráfico
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(2, 2)
    # Removing ticks and spines enables you to get the figure only with table
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    #configuración de formato condicional por columna
    def formatocondicional(vectorcolumna,ubicacion,cond2):
        vals =vectorcolumna
        normal = cm.colors.Normalize(vals.min()[0], vals.max()[0])
        if cond2==1:
            bcmap2 = plt.cm.spring(normal(vals))
        if cond2==2:
            bcmap2 = plt.cm.Spectral(normal(vals))
        for idx, bb in enumerate(bcmap2):
            tabla[(idx+1, ubicacion)].set_facecolor(bb) 
    formatocondicional(vPC,5,1)
    formatocondicional(vFAR,7,1)
    formatocondicional(vPOD,8,1) 
def g3(df,ncv1,ncv2,ncv3,lab1,lab2,lab3,t1):
    fig = plt.figure(figsize=(14,4)) #Creación de gráfico y tamaño
    ax1 = fig.add_subplot(111) #var2
    #Graficas 
    p1, = ax1.plot(df.fecha,df[ncv1],label=lab1)
    p2, = ax1.plot(df.fecha,df[ncv2],label=lab2)  
    p3, = ax1.plot(df.fecha,df[ncv3],label=lab3,color="black")    
    #Configuración de titulos y labels
    ax1.set_title(t1,loc="center",fontsize=12)
    ax1.set_ylabel("Precipitación [mm/día]", fontsize=12)   
    ax1.set_xlabel("TL (año - mes - día)",fontsize=12)
    #Otras configuraciones
    ax1.minorticks_on()
    ax1.legend(handles=[p1,p2,p3],loc="upper left")
    plt.grid()
#v,ncf,ncv1,ncv2=datos8,"fecha","SonsonWRF","Sonson"
def QMGAMMA(v,ncf,ncv1,ncv2):
    df1=v[v[ncf]>='2020-02-01 00:00:00'][v[ncf]<='2021-02-28 23:59:00']
    df2=v[v[ncf]>'2021-02-28 23:59:00'][v[ncv1].notnull()][v[ncv2].notnull()]
    biasCorreccion = BiasCorrection(df1[ncv2],df1[ncv1],df2[ncv1])
    df4 = biasCorreccion.correct(method='gamma_mapping')
    df3=pd.concat([df2[ncf],df2[ncv2],df2[ncv1],df4],axis=1)
    df3.columns=["fecha","obs","model","gamma"]
    df3=df3.set_index("fecha")
    df3=df3.resample('D').agg(pd.Series.mean,skipna=True)
    df3=df3.reset_index() 
    df3=P1FECHAS(df3,"fecha")
    return(df3)

###############################################################################
# 3. INFORMACIÓN DE ENTRADA Y ACLARACIONES
# 3.1 Direcciones de entrada
ent1=r"/media/luisa/Compartido/Luisa/8_SAMA/1proyectos/7Fechas/"
sal1=r"/media/luisa/Compartido/Luisa/9_TESIS/" #CSV
sal2=r"/media/luisa/Compartido/Luisa/8_SAMA/1proyectos/7Fechas/Otros/Resultados1/"# Sin corregir valores negativos
sal3=r"/media/luisa/Compartido/Luisa/8_SAMA/1proyectos/7Fechas/Otros/Resultados2/"

salida=sal2
# 3.2 Vectores con datos
Columnas = ["n","A","B","C","D","PC","Bias","FAR","POD"]
letras=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t",
        "u","v","w","x","y","z"]
Municipios=["Anza","Apartadó","Cáceres","Cisneros","Jericó","Puerto Triunfo",
            "Salgar","San Pedro","Sonsón"]
month=[1,2,3,4,5,6,7,8,9,10,11,12] #meses
d=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]#días
h=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23] #horas
###############################################################################
# 4.  PROCESOS
C1 = os.listdir(ent1) #Ficheros de la carpeta del proyecto 7
#-----------------------------------------------------------------------------#
# 4.1 FECHAS
datos=pd.read_csv(ent1+C1[0],names=["fecha"])
datos["fecha"]=pd.to_datetime(datos["fecha"])
#-----------------------------------------------------------------------------#
# 4.2 GFS VS GPM
print("1. Comparación :"+C1[1]) #Subtitulo
GFS=glob.glob(ent1+C1[1]+"/GFS*") #Vector de ficheros GFS
GPM=glob.glob(ent1+C1[1]+"/GPM*") #Vector de ficheros GPM
datos1=pd.concat([datos.fecha],axis=1)
m1=IngesarSeriesTiempo(GPM,"GPM")
m2=IngesarSeriesTiempo(GFS,"GFS")
datos1=pd.concat([datos1,m1,m2],axis=1)
#-----------------------------------------------------------------------------#
# 4.3 WRF VS GPM
print("2. Comparación :"+C1[6]) #Subtitulo
WRF=glob.glob(ent1+C1[6]+"/WRF*") #Vector de ficheros GFS
GPM=glob.glob(ent1+C1[6]+"/GPM*") #Vector de ficheros GPM
datos2=pd.concat([datos.fecha],axis=1)
m1=IngesarSeriesTiempo(GPM,"GPM")
m2=IngesarSeriesTiempo(WRF,"WRF")
datos2=pd.concat([datos2,m1,m2],axis=1)
#-----------------------------------------------------------------------------#
# 4.4 modelos vs observaciones
#---------------------------------------#
# 4.4.1 Observaciones
print("4.4.1 Observaciones")
OBS=pd.read_csv(sal1+"CSV/D2.csv",usecols=["fecha","obs","obs3"])
OBS=P1FECHAS(OBS,"fecha")
datos3=[]
for index, row in tqdm(datos.iterrows()):
    #print(row["fecha"])
    m=OBS.index[OBS.fecha==row["fecha"]].tolist()[0]
    m1=OBS["fecha"][m],OBS["obs"][m],OBS["obs3"][m]
    datos3.append(m1)
datos3=pd.DataFrame(datos3,columns=["fecha","Sonson","Rionegro"])
#---------------------------------------#
# 4.4.2 Modelos
print("4.4.2 Modelos")
#GFSvsGPM
GFS=glob.glob(ent1+C1[5]+"/GFS**GFSgrid.txt")
GPM=glob.glob(ent1+C1[5]+"/GPM**GFSgrid.txt")
datos4=pd.concat([datos.fecha],axis=1)
m1=IngesarSeriesTiempo(GPM,"GPM")
m2=IngesarSeriesTiempo(GFS,"GFS")
datos4=pd.concat([datos4,m1,m2],axis=1)


#WRFvsGPM
WRF=glob.glob(ent1+C1[5]+"/WRF**GPMgrid.txt")
GPM=glob.glob(ent1+C1[5]+"/GPM**GPMgrid.txt")
datos5=pd.concat([datos.fecha],axis=1)
m1=IngesarSeriesTiempo(GPM,"GPM")
m2=IngesarSeriesTiempo(WRF,"WRF")
datos5=pd.concat([datos5,m1,m2],axis=1)
#---------------------------------------#
# 4.4.3 Concatenar
datos6=pd.concat([datos4,datos3.Sonson,datos3.Rionegro],axis=1)
datos7=pd.concat([datos5,datos3.Sonson,datos3.Rionegro],axis=1)
datos6.to_csv(ent1+"HistoricoGFS.csv")
datos7.to_csv(ent1+"HistoricoWRF.csv")
#-----------------------------------------------------------------------------#
# 4.5 TABLAS DE CONTINGENCIA
# 4.5.1 Corrección de valores negativos
df=datos1
def correccionTC_GPM(df):
    Columnas_Correccion=list(df.columns)
    Columnas_Correccion.remove("fecha")
    for i in Columnas_Correccion:
        df[i][df[i]<0]=0.0
        #print(df[i][df[i]<0])
    return(df)
#datos1=correccionTC_GPM(datos1)
#datos2=correccionTC_GPM(datos2)
#datos6=correccionTC_GPM(datos6)
#datos7=correccionTC_GPM(datos7)


TC1=VectoresFinales(datos1,"GPM","GFS") #GFS vs GPM
TC2=VectoresFinales(datos2,"GPM","WRF") #GFS vs GPM
TC3=VectoresFinalesOBS(datos6,"GPM","GFS")
TC4=VectoresFinalesOBS(datos7,"GPM","WRF")



#-----------------------------------------------------------------------------#
# 4.6 GRÁFICAS DE LAS TABLAS DE CONTINGENCIA

fig = plt.figure(figsize=(10,5)) # Parametrizaciones 1
g1(TC1,1,1,1,fig,"Comparación GFS vs GPM","",0) 
plt.savefig(salida+"ComGFSvsGPM.png", bbox_inches='tight')
fig = plt.figure(figsize=(10,5)) # Parametrizaciones 1
g1(TC2,1,1,1,fig,"Comparación WFR vs GPM","",0) 
plt.savefig(salida+"ComWRFvsGPM.png", bbox_inches='tight')


fig = plt.figure(figsize=(10,5)) # Parametrizaciones 1
g2(TC3,1,1,1,fig,"Comparación GFS vs GPM vs OBS","",0,"GPM","GFS")
plt.savefig(salida+"ComGFSvsGPMvsOBS.png", bbox_inches='tight')
 
fig = plt.figure(figsize=(10,5)) # Parametrizaciones 1
g2(TC4,1,1,1,fig,"Comparación WRF vs GPM vs OBS","",0,"GPM","WRF") 
plt.savefig(salida+"ComWRFvsGPMvsOBS.png", bbox_inches='tight')

#-----------------------------------------------------------------------------#
# 4.7 BANCO DE LA REPUBLICA 
# Graficos, bias correction y validación
# Datos 6 tiene los municipios con GFS 
# Datos 7 tiene los municipios con WRF
#PASO1: Se agrupan los datos a analizar
datos8 = pd.concat([datos6.fecha,datos6.SonsonGFS,datos7.SonsonWRF,datos7.Sonson],axis=1)
datos8=datos8.set_index("fecha")
datos8=datos8.resample('D').agg(pd.Series.mean,skipna=True)
datos8=datos8.reset_index() 
#PASO2: Procesamiento de las fechas.
datos8 = P1FECHAS(datos8,"fecha")
y= list(datos8.year.unique())
y.sort()
#PASO3: Corrección del sesgo en el modelo WRF
#Datos9 es el vector con la corrección de sesgo
datos9 = QMGAMMA(datos8,"fecha","SonsonWRF","Sonson") #Corrección de sesgo
#PASO4: Tablas de contingencia
TC = vectorfinal2(datos9)
#PASO5: Gráficas
#df1 y df2 se ubican en octubre de 2021
df1=datos8[datos8.year==2021][datos8.month==8]
df2=datos9[datos9.year==2021][datos9.month==8]
df1=df1.reset_index(drop=True)
df2=df2.reset_index(drop=True)
df3=pd.concat([df1.fecha,df1.SonsonGFS,df2.gamma,df2.obs],axis=1)

g3(df1,"SonsonGFS","SonsonWRF","Sonson","GFS","WRF","OBS (SIATA - 199)",
   "Comparativo de acumulados de precipitación \n Modelos vs Observaciones\n Municipio de Sonsón ")
g3(df2,"gamma","model","obs","GAMMA_WRF","WRF","OBS (SIATA - 199)",
   "Comparativo de acumulados de precipitación\n Bias_correction \n Municipio de Sonsón ")
g3(df3,"gamma","SonsonGFS","obs","GFS","WRFBC","OBS (SIATA - 199)",
   "Comparativo de acumulados de precipitación\n Bias_correction \n Municipio de Sonsón ")

def g4(df,n1,n2,n3,cont,t1):
    def vector5(df,maxi,posicion):
        df1=[]
        for i in range(0,maxi):
            df1.append(df[i][posicion])
        df1=pd.DataFrame(df1)
        return(df1)
    #Información de entrada 
    filas=["model","gamma"]
    columnas = ["n","A","B","C","D","PC","Bias","FAR","POD","FAR/POD","POD/FAR"]    
    #Creación de vectores
    #PC, FAR y POD
    vPC=vector5(df,len(filas),5)
    vFAR=vector5(df,len(filas),7)
    vPOD=vector5(df,len(filas),8)
    #Gráfico
    #Ubicación del Subplot "Sub grafico"
    ax = fig.add_subplot(n1,n2,n3)
    #Titulos y labels de ejes
    ax.set_title(letras[cont]+") "+t1,fontsize=15,loc="left")
    #Gráfico
    tabla = ax.table(df,colWidths=[0.05] * len(columnas),rowLabels=filas, colLabels=columnas,
                     rowColours=["#ada587"]*len(filas),colColours=["#ada587"]*len(columnas),
                     cellLoc ='center', loc ="center")
    #Configuraciones del tamaño del gráfico
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(2, 2)
    # Removing ticks and spines enables you to get the figure only with table
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    #configuración de formato condicional por columna
    def formatocondicional(vectorcolumna,ubicacion,cond2):
        vals =vectorcolumna
        normal = cm.colors.Normalize(vals.min()[0], vals.max()[0])
        if cond2==1:
            bcmap2 = plt.cm.spring(normal(vals))
        if cond2==2:
            bcmap2 = plt.cm.Spectral(normal(vals))
        for idx, bb in enumerate(bcmap2):
            tabla[(idx+1, ubicacion)].set_facecolor(bb) 
    #formatocondicional(vPC,5,1)
    #formatocondicional(vFAR,7,1)
    #formatocondicional(vPOD,8,1)
 
fig = plt.figure(figsize=(10,5)) # Parametrizaciones 1
g4(TC,1,1,1,0,"")



######### Almacenar valores de tablas de contingencia############################
df,nombre=TC1,"TC1_GPMvsGFS"
def convert_to_csv(df,nombre):
    df=pd.DataFrame(df,columns=[Columnas+["FAR/POD","POD/FAR"]])
    df.to_csv(salida+nombre+".csv")

convert_to_csv(TC1,"TC1_GPMvsGFS")
convert_to_csv(TC2,"TC2_GPMvsWRF")
convert_to_csv(TC3,"TC3_GPMvsGFSvsOBS")
convert_to_csv(TC4,"TC4_GPMvsWRFvsOBS")




