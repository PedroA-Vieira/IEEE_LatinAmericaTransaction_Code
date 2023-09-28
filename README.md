# IEEE_LatinAmericaTransaction_Code
Files and programs used for studies and analysis of the article published in the IEEE Latin America Transactions magazine for the paper "Statistics, Coverage, and Improvement of Modelling via ANN of Radio Mobile Signal in Vegetated Channel in the 700-4000 MHz Band".

![GraphicalAbstractForPaperEnglish_v1a](https://github.com/PedroA-Vieira/IEEE_LatinAmericaTransaction_Code/assets/67390115/64812f0b-bffc-4974-af8b-2f634804da83)

# Instructions for running the programs
I - General observations

* The programs were written in Python version 3.7, the IDE Spyder version 5.3.3 was used and a specific virtual environment was created to execute the programs
* The main necessary libraries (dependencies) are right at the beginning of the program
* The ArquivosDataFrame folder contains the pre-processed data used to carry out the analyzes and tests as weel systens data
* Within programs, the search path for data files must be adjusted.

II - Programs for processing the measured signal

The program: SinalVegetacao_GraficosSinaisProcessados_v3.py has as input the files compressed in the folder: ArquivosProcessamento and the file with the system data: dadosDoSistema_v1.xlsx.
Once unzipped, the files will be in .xlsx format.

III - Programs for studying the machine learning model.

The programs:
* SinalVegetacao_MLP_Keras_v14 and SinalVegetacao_MLP_Sklearn_v13.py
are essentially the same. The first uses Keras library and the second SkLearn library, but both building multilayer perceptron neural network (RNA-MLP) architecture.
The inputs are not the same or the results equivalent.
