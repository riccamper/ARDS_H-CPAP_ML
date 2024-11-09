# FILE DI UTILITIES CON FUNZIONI E METODI
#####

# INIZIALIZZAZIONE
import math
import random
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import ParameterGrid, train_test_split
#from impyute.imputation.cs import mice
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, OneHotEncoder, StandardScaler
#from tabulate import tabulate
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, recall_score, accuracy_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from joblib import dump, load

# RANDOM STATE
seed = 44
def get_seed():
	return seed

# AUDIO
#from vscode_audio import Audio
#sound = []
#def SoundNotification(sec):
#	global sound
#	sr = 22050 # sample rate
#	T = sec    # seconds
#	t = np.linspace(0, T, int(T*sr), endpoint=False)
#	x = (0.33*np.sin(2*np.pi*523*t) + 0.33*np.sin(2*np.pi*659*t) + 0.33*np.sin(2*np.pi*784*t))*np.sin(np.pi*(1/T)*t*10)
#	#sound = ipd.Audio(x, rate=sr, autoplay=True)
#	sound = Audio(x, sr)
#	return sound
#def SineWave(sec=2):
#	sound = SoundNotification(sec)
#	return sound


# LETTURA DATASET
def read_dataset(filename):
	dataset = pd.read_csv(filename, sep=';', index_col=0, encoding='utf-8')
	return dataset


# ELIMINAZIONE RECORD TROPPO VECCHI
def elimina_record_troppo_vecchi(dataset, par_data_ricovero='2013-03', par_cpap_ok=1):
	print("\n* Eliminazione record troppo vecchi")
	print("\t- record con data_ricovero <", par_data_ricovero)
	#print("\t- record con CPAP_OK ==", par_cpap_ok)
	dataset_len = len(dataset)
	print("\t- dimensione dataset:", dataset_len)
	#dataset = dataset.drop(dataset[(dataset['data_ricovero'] < par_data_ricovero) & (dataset['CPAP_OK'] == par_cpap_ok)].index)
	dataset = dataset.drop(dataset[(dataset['data_ricovero'] < par_data_ricovero)].index)
	print("\t- record eliminati:", dataset_len - len(dataset))
	print("\t- dimensione nuovo dataset:", len(dataset))
	return dataset

# ELIMINAZIONE RECORD CON PF NULLO
def elimina_record_feature_nullo(dataset, feature="PaO2/FiO2 Ratio", par_cpap_ok=1):
	print("\n* Eliminazione record", feature, "nullo")
	print("\t- record con", feature, "= NaN")
	print("\t- record con CPAP_OK ==", par_cpap_ok)
	dataset_len = len(dataset)
	print("\t- dimensione dataset:", dataset_len)
	#dataset = dataset.drop(dataset[(pd.isna(dataset[feature])) & (dataset['CPAP_OK'] == par_cpap_ok)].index)
	dataset = dataset.drop(dataset[(pd.isna(dataset[feature]))].index)
	print("\t- record eliminati:", dataset_len - len(dataset))
	print("\t- dimensione nuovo dataset:", len(dataset))
	return dataset

# FEATURE PER RICONOSCERE QUALI SONO SENZA ABG E CLASSE POSITIVA
def feature_no_abg_e_classe_positiva(dataset, feature="No ABG"):
	dataset[feature] = pd.Series(0, index=dataset.index).mask(pd.isna(dataset['PaO2/FiO2 Ratio']), 1)
	return dataset

# ELIMINAZIONE RECORD
def elimina_record(dataset, parametro='diagnosi_descr', valore='Insufficienza respiratoria acuta e cronica'):
	print("\n* Eliminazione record")
	print("\t- record con", parametro, "=", valore)
	dataset_len = len(dataset)
	print("\t- dimensione dataset:", dataset_len)
	dataset = dataset.drop(dataset[(dataset[parametro] == valore)].index)
	print("\t- record eliminati:", dataset_len - len(dataset))
	print("\t- dimensione nuovo dataset:", len(dataset))
	return dataset

# ELIMINAZIONE RECORD
def elimina_record_nan(dataset, parametro='diagnosi_descr'):
	print("\n* Eliminazione record")
	print("\t- record con", parametro, "= NaN")
	dataset_len = len(dataset)
	print("\t- dimensione dataset:", dataset_len)
	dataset = dataset.dropna(subset=[parametro])
	#dataset = dataset.drop(dataset[(dataset[parametro] == np.nan)].index)
	print("\t- record eliminati:", dataset_len - len(dataset))
	print("\t- dimensione nuovo dataset:", len(dataset))
	return dataset

# ELIMINAZIONE RECORD
def elimina_record_non_ards(dataset, parametro='PaO2/FiO2 Ratio (worst)'):
	print("\n* Eliminazione record non ARDS")
	print("\t- record con", parametro, "> 300")
	dataset_len = len(dataset)
	print("\t- dimensione dataset:", dataset_len)
	dataset = dataset.drop(dataset[(dataset[parametro] > 300)].index)
	#dataset = dataset.drop(dataset[(dataset[parametro] == np.nan)].index)
	print("\t- record eliminati:", dataset_len - len(dataset))
	print("\t- dimensione nuovo dataset:", len(dataset))
	return dataset

# ELIMINAZIONE RECORD
def elimina_record_non_ards_finale(dataset, parametro='PaO2/FiO2 Ratio'):
	print("\n* Eliminazione record non ARDS")
	print("\t- record con", parametro, "> 300")
	dataset_len = len(dataset)
	print("\t- dimensione dataset:", dataset_len)
	dataset = dataset.drop(dataset[(dataset[parametro] > 300)].index)
	#dataset = dataset.drop(dataset[(dataset[parametro] == np.nan)].index)
	print("\t- record eliminati:", dataset_len - len(dataset))
	print("\t- dimensione nuovo dataset:", len(dataset))
	return dataset

# MARKING RECORD PAZIENTE DUPLICATO
def mark_record_paziente_duplicato(dataset, parametro="cod_individuale", keep="last"):
	print("\n* Marking record duplicati")
	dataset_len = len(dataset)
	print("\t- dimensione dataset:", dataset_len)
	#dataset = dataset.drop_duplicates(keep=keep, subset=[parametro], inplace=False)
	dataset["Duplicato"] = dataset.duplicated(subset=parametro, keep=keep)
	#print("\t- record eliminati:", dataset_len - len(dataset))
	#print("\t- dimensione nuovo dataset:", len(dataset))
	return dataset

# ELIMINAZIONE RECORD PAZIENTE DUPLICATO
def elimina_record_paziente_duplicato(features_train, features_test, labels_train, labels_test, parametro="cod_individuale", keep="first"):
	print("\n* Eliminazione record duplicati")
	features_train_len, features_test_len = len(features_train), len(features_test)
	print("\t- dimensione datasets:", features_train_len, features_test_len)
	labels_train = labels_train.drop(labels_train[(features_train["Duplicato"] == 1)].index)
	features_train = features_train.drop(features_train[(features_train["Duplicato"] == 1)].index)
	labels_test = labels_test.drop(labels_test[(features_test["Duplicato"] == 1)].index)
	features_test = features_test.drop(features_test[(features_test["Duplicato"] == 1)].index)
	print("\t- record eliminati:", features_train_len - len(features_train),  features_test_len - len(features_test))
	print("\t- dimensione nuovi datasets:", len(features_train), len(features_test))
	return features_train, features_test, labels_train, labels_test


# DROP FEATURES
def drop_features(dataset, columns, silent=False):
	if not silent:
		print("\n* Drop features")
	n_columns = len(dataset.columns)
	if not silent:
		print("\t- numero features dataset:", n_columns)
	dataset = dataset.drop(columns, axis=1)
	if not silent:
		print("\t- numero features eliminate:", n_columns - len(dataset.columns))
		print("\t- numero features rimanenti:", len(dataset.columns))
		print("\t- features eliminate:", columns)
		print("\t- features rimanenti:", dataset.columns.tolist())
	return dataset


# ESTRAZIONE FEATURES E LABELS
def estrai_features_e_labels(dataset):
	print("\n* Estrazione features e labels")
	dataset = dataset.rename(columns={"CPAP_OK": "H-CPAP failure"})
	print("\t- feature CPAP_OK rinominata in CPAP_OK")
	labels = dataset['H-CPAP failure']
	labels = labels.astype(bool)
	labels = ~labels # negazione
	labels = labels.astype(int)
	print("\t- shape dataset labels:", labels.shape)
	features = dataset.drop('H-CPAP failure', axis = 1)
	print("\t- shape dataset features:", features.shape)
	return [features, labels]


# TRASFORMAZIONE DI SERIE [] IN NUMERI
def trasforma_serie_in_numeri(features_copy, punti=3, method='reglin', last=0):
	print("\n* Trasformazione di serie [] in numeri")
	features = features_copy.copy()
	plot = 0
	plot_feature, plot_arr_temp, plot_arr_reg_temp, plot_value, plot_label = '', [], [], 0, ''
	features_arr = []
	for feature in features.columns:
			for i, value in features[feature].items():
				if type(value) == str and value[0] == '[' and value[len(value)-1] == ']':
					value_temp = value
					value_temp = value_temp.replace('[', '')
					value_temp = value_temp.replace(']', '')
					arr_temp = value_temp.split(',')
					arr_temp = [float(x) for x in arr_temp]
					if method == 'reglin':
						value_temp = linear_regression(arr_temp, punti, last)
					elif method == 'avg':
						value_temp = average(arr_temp, punti, last, weighted=False)
					elif method == 'wavg':
						value_temp = average(arr_temp, punti, last, weighted=True)
					elif method == 'last':
						value_temp = arr_temp[len(arr_temp)-1]
					elif method == 'first':
						value_temp = arr_temp[0]
					elif method == 'min':
						value_temp = min(arr_temp)
					elif method == 'max':
						value_temp = max(arr_temp)
					features[feature][i] = float(value_temp)
					# PLOT
					if plot <= 5  and len(arr_temp) > 10 and feature == 'D-Dimer':
						if method == 'reglin':
							arr_reg_temp = [linear_regression(arr_temp, punti, punti - j - 1) for j in range(punti)]
							plot_feature, plot_arr_temp, plot_arr_reg_temp, plot_value, plot_label = feature, arr_temp, arr_reg_temp, features[feature][i], 'Linear regression'
						elif method == 'avg':
							arr_reg_temp = [average(arr_temp, punti, punti - j - 1, weighted=False) for j in range(punti)]
							plot_feature, plot_arr_temp, plot_arr_reg_temp, plot_value, plot_label = feature, arr_temp, arr_reg_temp, features[feature][i], 'Average'
						elif method == 'wavg':
							arr_reg_temp = [average(arr_temp, punti, punti - j - 1, weighted=True) for j in range(punti)]
							plot_feature, plot_arr_temp, plot_arr_reg_temp, plot_value, plot_label = feature, arr_temp, arr_reg_temp, features[feature][i], 'Weighted average'
						elif method == 'last':
							plot_feature, plot_arr_temp, plot_arr_reg_temp, plot_value, plot_label = feature, arr_temp, [], features[feature][i], 'Average'
						plot = plot + 1
					# APPEND TO RESULTS
					if feature not in features_arr:
						features_arr.append(feature)
						print("\t- features trasformate:", features_arr, end = '\r')
	print('')
	#plot_linear_regression(plot_feature, plot_arr_temp, plot_arr_reg_temp, plot_value, plot_label)
	return features

# REGRESSIONE LINEARE		
def linear_regression(arr, maxm, last):
	if len(arr) == 1:
		return arr[0]
	elif maxm <= 1:
		return arr[len(arr)-1]
	else:
		maxm = np.minimum(maxm, len(arr))
		start = len(arr) - maxm
		X_l_reg = []
		y_l_reg = []
		for i in np.arange(start, len(arr)):
			X_l_reg.append(i - start)
			y_l_reg.append([arr[i]])
		l_reg = LinearRegression().fit(np.array(X_l_reg).reshape(-1, 1), y_l_reg)
		return l_reg.predict( [[maxm-1-last]] )[0]

# MEDIA PESATA	
def average(arr, maxm, last, weighted=False):
	if len(arr) == 1:
		return arr[0]
	elif maxm <= 1:
		return arr[len(arr)-1]
	else:
		maxm = np.minimum(maxm, len(arr))
		start = len(arr) - maxm
		avg = 0
		w = []
		w_max_sum = 0
		for i in np.arange(len(arr) - start - last):
			if weighted == False:
				w.append(1)
			else:
				w.append(10**(i + 1))
		w_max_sum = np.max(sum(w))
		for i in np.arange(len(arr) - start - last):
			w[i] = w[i] / w_max_sum
		for i in np.arange(start, len(arr) - last):
			avg = avg + arr[i] * w[i - start]
		return avg

# PLOT BEFORE LINEAR REGRESSION
def plot_linear_regression(feature, arr, arr_reg, value, lab):
	plt.figure(figsize=(5, 2))
	plt.title(feature)
	y, x = arr, np.arange(0, len(arr))
	plt.plot(x, y, label='Time-sorted vector', color='#5C8A99') # Colore opposto: #24373D
	y2, x2 = arr_reg, np.arange(0, len(arr_reg))
	if len(arr_reg) != 0:
		plt.plot(len(x) - len(x2) + x2, y2, label=lab, color='red', linewidth='2')
	plt.plot(len(x) - 1, value, marker='o', label='Chosen value', color='red', linewidth='2')
	#plt.axvline(x=original_features_train_shape[0], color='red', linestyle='--')
	#plt.axhline(y=np.mean(train), color='red', linestyle='--', label='Mean = %0.2f' % np.mean(train))
	plt.ylabel("Value")
	plt.legend(loc='best')
	plt.grid()
	plt.tight_layout()
	plt.show()


# IMPUTAZIONE FEATURES FiO2, pO2, RAPPORTO pO2/FiO2
def imputa_fi02_po2_rapporto_po2_fio2(features):
	print("\n* Imputazione features FiO2, pO2, RAPPORTO pO2/FiO2")
	anomalie = 0
	for index, row in features.iterrows():
		if math.isnan(row['PaO2']) and not math.isnan(row['FiO2']) and not math.isnan(row['PaO2/FiO2 Ratio']):
			features['PaO2'][index] = float( ( features['PaO2/FiO2 Ratio'][index] * features['FiO2'][index] ) / 100 )
			anomalie+=1
		elif not math.isnan(row['PaO2']) and math.isnan(row['FiO2']) and not math.isnan(row['PaO2/FiO2 Ratio']):
			features['FiO2'][index] = float( ( features['PaO2'][index] / features['PaO2/FiO2 Ratio'][index] ) * 100 )
			anomalie+=1
		elif not math.isnan(row['PaO2']) and not math.isnan(row['FiO2']) and math.isnan(row['PaO2/FiO2 Ratio']):
			features['PaO2/FiO2 Ratio'][index] = float( ( features['PaO2'][index] / features['FiO2'][index] ) * 100 )
			anomalie+=1
	print("\t- anomalie trovate e corrette sui dati:", anomalie)
	return features


# IMPUTAZIONE FEATURES FiO2, pO2, RAPPORTO pO2/FiO2
def imputa_rapporto_po2_fio2(features):
	print("\n* Imputazione features RAPPORTO pO2/FiO2")
	for index, row in features.iterrows():
		features['PaO2/FiO2 Ratio'][index] = float( ( features['PaO2'][index] / features['FiO2'][index] ) * 100 )
	return features


# ISPEZIONE MANUALE DELLE FEATURES
#def ispeziona_features(features):
#	print("\n* Ispezione manuale delle features")
#	anomalie = 0
#	for index, row in features.iterrows():
#		
#		if math.isnan(row['PaO2']) and not math.isnan(row['FiO2']) and not math.isnan(row['PaO2/FiO2 Ratio']):
#			features['PaO2'][index] = float( ( features['PaO2/FiO2 Ratio'][index] * features['FiO2'][index] ) / 100 )
#			anomalie+=1
#		elif not math.isnan(row['PaO2']) and math.isnan(row['FiO2']) and not math.isnan(row['PaO2/FiO2 Ratio']):
#			features['FiO2'][index] = float( ( features['PaO2'][index] / features['PaO2/FiO2 Ratio'][index] ) * 100 )
#			anomalie+=1
#		elif not math.isnan(row['PaO2']) and not math.isnan(row['FiO2']) and math.isnan(row['PaO2/FiO2 Ratio']):
#			features['PaO2/FiO2 Ratio'][index] = float( ( features['PaO2'][index] / features['FiO2'][index] ) * 100 )
#			anomalie+=1
#	print("\t- anomalie trovate e corrette sui dati:", anomalie)
#	return features


# RIPRISTINO VALORI NULLI FUMATORE/DIABETICO
def ripristino_valori_nulli_fumatore_diabetico(features):
	print("\n* Ripristino valori nulli per features fumatore e diabetico")
	anomalie = 0
	for index, row in features.iterrows():
		if math.isnan(row['Hypertension']):
			features['Smoker'][index] = np.nan
			features['Diabetes'][index] = np.nan
			anomalie+=1
	print("\t- anomalie trovate e corrette sui dati:", anomalie)
	return features


# CONTO MISSING VALUES E SCARTO FEATURES
def conto_missing_values_e_scarto_features(features, percentuale=0.4, features_to_keep=['D_DIMERO'], silent=False):
	print("\n* Conto missing values e scarto features con piu' del", percentuale * 100, "% di missing values")
	features_drop_arr = []
	features_keep_arr = []
	to_plot = {}
	to_not_plot = ["Duplicato", "No ABG"]
	for feature in features.columns:
		# TO PLOT THE FEATURES
		if feature not in to_not_plot:
			to_plot[feature] = (features[feature].isnull().sum() / len(features[feature]))
		# CHECK
		if features[feature].isnull().sum() / len(features[feature]) >= percentuale and feature not in features_to_keep:
			features_drop_arr.append("%s: %.2f" % (feature, (features[feature].isnull().sum() / len(features[feature])) ) )
			features = features.drop([feature], axis=1)
		else:
			features_keep_arr.append("%s: %.2f" % (feature, (features[feature].isnull().sum() / len(features[feature])) ) )
	print("\t- numero features eliminate:", len(features_drop_arr))
	print("\t- numero features mantenute:", len(features_keep_arr))
	print("\t- features eliminate:", features_drop_arr)
	if not silent:
		print("\t- features mantenute:", features_keep_arr)
		# PLOT
		df = pd.DataFrame.from_dict(to_plot, orient='index').\
		rename(columns={0: 'Percentage of missing values'}).\
		sort_values(by='Percentage of missing values', ascending=False)
		fig, ax = plt.subplots(figsize=(6,10))  # Adjust the figure size
		for i, val in enumerate(df['Percentage of missing values']):
			color = '#5C8A99' if val <= percentuale else '#24373D'
			ax.barh(df.index[i], val, color=color)  # Use barh for horizontal bars
		plt.grid(True)
		plt.xlim(0,0.5)  # Adjust the x-limits of the plot
		plt.ylim(-0.5, len(df.index)-0.5)  # Adjust the y-limits of the plot
		plt.axvline(x=percentuale, color='r', linestyle='-')  # Draw a vertical line
		# Percentages
		vals = ax.get_xticks()
		ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])  # Set the x-tick labels
		# Add labels for the two categories
		low_patch = mpatches.Patch(color='#5C8A99', label='Features to keep')
		high_patch = mpatches.Patch(color='#24373D', label='Features discarded')
		line = mpatches.Patch(color='r', label='Threshold')
		plt.legend(handles=[low_patch, high_patch, line])
		plt.show()
	return features


# CONVERSIONE DATAFRAME IN FLOAT
def converto_dataframe_in_float(dataframe, bools_col=[]):
	print("\n* Conversione dataframe in float")
	dataframe = dataframe.astype(object)
	i = 0
	for column in dataframe.columns:
		if column not in bools_col:
			dataframe[column] = dataframe[column].astype(float)
			i+=1
		else:
			# ARROTONDAMENTO > 0.5
			for i, value in dataframe[column].items():
				if value > 0.5:
					dataframe[column][i] = float(1.0)
				else:
					dataframe[column][i] = float(0.0)
	print("\t- numero colonne convertite in float:", i)
	return dataframe


# RICERCA E SOSTITUZIONE OUTLIERS
def ricerco_e_sostituisco_outliers(features, hi=0.97, low=0.03, silent=False):
	print("\n* Ricerca e sostituzione outliers (valore > ", hi * 100, "percentile || valore < ", low * 100, "percentile)")
	outliers = [0, 0]
	features_arr = []
	to_plot = {}
	for feature in features.columns:
		HIquant = np.quantile(features[feature].dropna(), 0.97)
		LOquant = np.quantile(features[feature].dropna(), 0.03)
		out = [0, 0]
		if not is_col_bool(features[feature]):
			for i, value in features[feature].items():
				if value > HIquant:
					features[feature][i] = HIquant
					outliers[0] = outliers[0] + 1
					out[0] = out[0] + 1
				elif value < LOquant:
					features[feature][i] = LOquant
					outliers[1] = outliers[1] + 1
					out[1] = out[1] + 1
			features_arr.append("%s: %d positivi > %.02f, %d negativi < %.02f" % (feature, out[0], HIquant, out[1], LOquant) )
		# PLOT
		if not is_col_bool(features[feature]):
			to_plot[feature] = out[0] + out[1] #add the name/value pair
	if not silent:
		print("\t- outliers report:", features_arr)
		# PLOT
		pd.DataFrame.from_dict(to_plot, orient='index').\
		rename(columns={0: 'Number of outliers found'}).\
		sort_values(by='Number of outliers found', ascending=True).\
		plot(kind='bar', rot=90, figsize=(10,3), grid=True, color='#5C8A99')
		plt.show()
	print("\t-", outliers[0] + outliers[1], "outliers trovati,", outliers[0], "positivi,", outliers[1], "negativi")
	return features


# PLOT BOXPLOT
def plot_boxplot(features, feature):
	# PLOT
	features.boxplot(column=[feature], figsize=(4,3), grid=True, color='#5C8A99')
	plt.show()


# SE UNA COLONNA E' SOLO BOOLEANI
def is_col_bool(col):
	arr = np.sort( np.array([0, 1]) )
	unique = col.unique()
	unique = unique[np.logical_not(pd.isna(unique))]
	unique = unique.astype(float)
	unique = np.sort(unique)
	if np.array_equal(arr,unique):
		return True
	else:
		return False


# OTTENGO UNA LISTA DI SOLI INDICI DI COLONNE BOOLEANE
def get_cols_bool(features):
	print("\n* Ricerca di colonne booleane")
	features_arr = []
	for feature in features.columns:
		if is_col_bool(features[feature]):
			features_arr.append(feature)
			print("\t- features booleane:", features_arr, end = '\r')
	print('')
	return features_arr


# SHUFFLE FEATURES E LABELS
def shuffle_features_e_labels(features, labels):
	print("\n* Randomizzazione features e labels")
	features, labels = shuffle(features, labels, random_state=seed)
	features = features.reset_index(drop=True)
	labels = labels.reset_index(drop=True)
	return [features, labels]


# SPLIT DATASET IN TRAINING E TESTING SET
def split_dataset_in_training_e_testing_set(features, labels, test_size=0.15, shuffle=True, test_balanced=False):
	print("\n* Split in training e testing set")
	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size, random_state=seed, shuffle=True, stratify=labels)
	# Move rows to train set until 50/50 in test set
	if test_balanced:
		for feature_test_index, label_test_index in zip(features_test.index, labels_test.index):
			if labels_test.value_counts(normalize=True)[1] >= 0.5:
				break
			if labels_test.loc[label_test_index] == 0:
				features_train = features_train.append(features_test.loc[[feature_test_index]])
				labels_train = labels_train.append(labels_test.loc[[label_test_index]])
				features_test = features_test.drop(feature_test_index)
				labels_test = labels_test.drop(label_test_index)
	print("\t- record in testing set:", len(features_train), "features,", len(labels_train), "labels")
	print("\t- record in training set:", len(features_test), "features,", len(labels_test), "labels")
	return features_train, features_test, labels_train, labels_test


# SWAP NO ABG RECORDS
def swap_no_abg_records(features_train, features_test, labels_train, labels_test):
	print("\n* Swap record senza ABG dal test al train e viceversa")
	count = features_test["No ABG"].value_counts()[1]
	print("\t- record in testing set without ABG, positive class:", count)
	for feature_test_index, label_test_index in zip(features_test.index, labels_test.index):
		if 1 not in features_test["No ABG"].value_counts().index:
			break
		if features_test.loc[feature_test_index]["No ABG"] == 1 and labels_test.loc[label_test_index] == 1:
			features_train = features_train._append(features_test.loc[[feature_test_index]], ignore_index=True)
			labels_train = labels_train._append(labels_test.loc[[label_test_index]], ignore_index=True)
			features_test = features_test.drop(feature_test_index)
			labels_test = labels_test.drop(label_test_index)
	for features_train_index, label_train_index in zip(features_train.index, labels_train.index):
		if count == 0:
			break
		if features_train.loc[features_train_index, "No ABG"] == 0 and labels_train.loc[label_train_index] == 1 and random.randint(0, 1) == 0:
			features_test = features_test._append(features_train.loc[[features_train_index]], ignore_index=True)
			labels_test = labels_test._append(labels_train.loc[[label_train_index]], ignore_index=True)
			features_train = features_train.drop(features_train_index)
			labels_train = labels_train.drop(label_train_index)
			count-=1
	print("\t- record in testing set:", len(features_train), "features,", len(labels_train), "labels")
	print("\t- record in training set:", len(features_test), "features,", len(labels_test), "labels")
	return features_train, features_test, labels_train, labels_test


# SHUFFLE TRAINING SET E TESTING SET
def shuffle_training_set_e_testing_set(features_train, features_test, labels_train, labels_test):
	print("\n* Randomizzazione training set e testing set")
	features_train, labels_train = shuffle(features_train, labels_train, random_state=seed)
	features_train = features_train.reset_index(drop=True)
	labels_train = labels_train.reset_index(drop=True)
	features_test, labels_test = shuffle(features_test, labels_test, random_state=seed)
	features_test = features_test.reset_index(drop=True)
	labels_test = labels_test.reset_index(drop=True)
	return features_train, features_test, labels_train, labels_test


# IMPUTAZIONE VALORI MANCANTI
def imputo_valori_mancanti(features_train, features_test, min, max):
	print("\n* Imputazione valori mancanti")
	imputer = IterativeImputer(
		#estimator=LinearRegression(n_jobs=-1),
		#estimator=RandomForestRegressor(n_jobs=-1),
		estimator=BayesianRidge(),
		missing_values=np.nan,
		sample_posterior=False,
		initial_strategy='mean',
		min_value=min,
		max_value=max,
		random_state=seed,
		max_iter=1000
	)
	imputer.fit(features_train) # FIT SOLO SU TRAINING SET
	features_train_old, features_test_old = features_train, features_test
	features_train = pd.DataFrame(imputer.transform(features_train), columns=features_train.columns, dtype=object)
	features_test = pd.DataFrame(imputer.transform(features_test), columns=features_test.columns, dtype=object)
	# DATAFRAMES "IMPUTED"
	for feature in features_train_old.columns:
		for i, value in features_train_old[feature].items():
			if pd.isna(value):
				features_train_old[feature][i] = 1
			else:
				features_train_old[feature][i] = 0
	for feature in features_test_old.columns:
		for i, value in features_test_old[feature].items():
			if pd.isna(value):
				features_test_old[feature][i] = 1
			else:
				features_test_old[feature][i] = 0
	return features_train, features_train_old, features_test, features_test_old


# IMPUTAZIONE VALORI MANCANTI (KNN)
def imputo_valori_mancanti_knn(features_train, features_test, k):
	print("\n* Imputazione valori mancanti (KNN)")
	features_train_old, features_test_old = features_train, features_test
	scaler = MinMaxScaler()
	scaler.fit(features_train.iloc[:, :])
	features_train.iloc[:, :] = scaler.transform(features_train.iloc[:, :])
	features_test.iloc[:, :] = scaler.transform(features_test.iloc[:, :])
	imputer = KNNImputer(
		missing_values=np.nan,
		n_neighbors=k,
		weights="uniform",
		metric="nan_euclidean",
	)
	imputer.fit(features_train) # FIT SOLO SU TRAINING SET
	features_train = pd.DataFrame(imputer.transform(features_train), columns=features_train.columns, dtype=object)
	features_test = pd.DataFrame(imputer.transform(features_test), columns=features_test.columns, dtype=object)
	features_train.iloc[:, :] = scaler.inverse_transform(features_train.iloc[:, :])
	features_test.iloc[:, :] = scaler.inverse_transform(features_test.iloc[:, :])
	# DATAFRAMES "IMPUTED"
	for feature in features_train_old.columns:
		for i, value in features_train_old[feature].items():
			if pd.isna(value):
				features_train_old[feature][i] = 1
			else:
				features_train_old[feature][i] = 0
	for feature in features_test_old.columns:
		for i, value in features_test_old[feature].items():
			if pd.isna(value):
				features_test_old[feature][i] = 1
			else:
				features_test_old[feature][i] = 0
	return features_train, features_train_old, features_test, features_test_old
	return features_train, features_test


# OVERSAMPLING (SMOTE)
def oversampling(features_train, labels_train, off=False, strategy=0.5):
	original_features_train_shape = features_train.shape
	if not off:
		print("\n* Oversampling (SMOTE)")
		#original_features_train_shape = features_train.shape
		sm = SMOTE(random_state=seed, sampling_strategy=strategy)
		features_oversample, labels_oversample = sm.fit_resample(features_train, labels_train)
		features_oversample = features_oversample
		print("\t- numero record nella classe minore aggiunti:", features_oversample.shape[0] - features_train.shape[0])
		print("\t- numero record nella classe minore correnti:", features_oversample.shape[0])
		features_train, labels_train = features_oversample, labels_oversample
	return features_train, labels_train, original_features_train_shape


# SCALING
def scaling(features_train, features_test, silent=False):
	if not silent:
		print("\n* Scaling con MinMaxScaler")
	scaler = MinMaxScaler()
	scaler.fit(features_train.iloc[:, :])
	features_train.iloc[:, :] = scaler.transform(features_train.iloc[:, :])
	features_test.iloc[:, :] = scaler.transform(features_test.iloc[:, :])
	return features_train, features_test, scaler


# STANDARDIZATION
def standardization(features_train, features_test):
	print("\n* Standardizzazione con StandardScaler")
	standardizer = StandardScaler()
	standardizer.fit(features_train.iloc[:, :])
	features_train.iloc[:, :] = standardizer.transform(features_train.iloc[:, :])
	features_test.iloc[:, :] = standardizer.transform(features_test.iloc[:, :])
	return features_train, features_test, standardizer


# DISCRETIZZAZIONE
def discretizing(features_train, features_test):
	print("\n* Discretizzazione con KBinsDiscretizer")
	discr = KBinsDiscretizer(n_bins=31, encode='ordinal', strategy='quantile', random_state=seed)
	discr.fit(features_train.iloc[:, :])
	features_train.iloc[:, :] = discr.transform(features_train.iloc[:, :])
	features_test.iloc[:, :] = discr.transform(features_test.iloc[:, :])
	return features_train, features_test, discr


# CALCOLA PRECISION-RECALL CURVE
def calcola_precision_recall_curve_old(model, features, labels):
	precisions, recalls, thresholds = precision_recall_curve(features, model.predict_proba(labels)[:, 1], pos_label=1)
	f1_scores_numerator = 2 * recalls * precisions
	f1_scores_denom = recalls + precisions
	f1_scores = np.divide(f1_scores_numerator, f1_scores_denom, out=np.zeros_like(f1_scores_denom), where=(f1_scores_denom != 0))
	return [precisions, recalls, f1_scores, thresholds]

def calcola_precision_recall_curve(labels_true, labels_pred):
	# Classe positiva
	accuracies, precisions, recalls, thresholds = precision_recall_curve(labels_true, labels_pred, pos_label=1)
	# ADD TO SKLEARN CODE "_ranking.py"
    #    fns = tps[-1] - tps
    #    tns = fps[-1] - fps
    #    accuracy = (tps + tns) / (tps + tns + fps + fns)
	f1_scores_numerator = 2 * recalls * precisions
	f1_scores_denom = recalls + precisions
	f1_scores = np.divide(f1_scores_numerator, f1_scores_denom, out=np.zeros_like(f1_scores_denom), where=(f1_scores_denom != 0))
	return [accuracies, precisions, recalls, f1_scores, thresholds]

# CALCOLA THRESHOLD DA PRECISION-RECALL CURVE
def calcola_precision_recall_curve_threshold(accuracies, precisions, recalls, f1_scores, thresholds):
	max_f1_score_index = np.argmax(f1_scores)
	max_f1_score = np.max(f1_scores)
	max_f1_thresh = thresholds[max_f1_score_index]
	return [max_f1_score_index, max_f1_score, max_f1_thresh]

# PLOT PRECISION-RECALL CURVE
def plot_precision_recall_curve(accuracies, precisions, recalls, f1_scores, thresholds, max_f1_score_index, max_f1_score, max_f1_thresh, old_max_f1_thresh=False):
	plt.figure(figsize=(2.5, 5))
	#plt.title("Precision, recall and F1-score as a function of the threshold")
	plt.plot(thresholds, precisions[:-1], "b-", linestyle='-.' if old_max_f1_thresh else '-', label="Precision")# = %0.2f" % precisions[max_f1_score_index])
	plt.plot(thresholds, recalls[:-1], "g-", linestyle='-.' if old_max_f1_thresh else '-', label="Recall")# = %0.2f" % recalls[max_f1_score_index])
	plt.plot(thresholds, f1_scores[:-1], "r-", linestyle='-.' if old_max_f1_thresh else '-', label="F1-score")# = %0.2f" % max_f1_score)
	plt.plot(thresholds, accuracies[:-1], "k-", linestyle='-.' if old_max_f1_thresh else '-', label="Accuracy")# = %0.2f" % max_f1_score)
	#plt.axvline(x=old_max_f1_thresh, color='red', linestyle='-.', label="Threshold")# = %0.2f" % old_max_f1_thresh)
	#if old_max_f1_thresh:
	#	plt.axvline(x=old_max_f1_thresh, color='red', label="Threshold = %0.2f" % old_max_f1_thresh)
	#else:
	if old_max_f1_thresh:
		plt.axvline(x=max_f1_thresh, color='red', linestyle='-.', label="Test threshold = %0.2f" % max_f1_thresh)
		plt.axvline(x=old_max_f1_thresh, color='red', label="CV threshold = %0.2f" % old_max_f1_thresh)
		plt.xlabel("Test decision threshold")
	else:
		plt.axvline(x=max_f1_thresh, color='red', label="CV threshold = %0.2f" % max_f1_thresh)
		plt.xlabel("CV decision threshold")
	plt.xlim([0.3, 0.7])
	plt.ylim([0.6, 1])
	plt.ylabel("Score")
	plt.legend(loc='best')
	plt.grid()


# CALCOLA ROC CURVE
def calcola_roc_curve(labels_true, labels_pred):
	fpr, tpr, thresholds = roc_curve(labels_true, labels_pred, pos_label=1)
	auc_scores = auc(fpr, tpr)
	gmeans = np.sqrt(tpr * (1-fpr))
	return [fpr, tpr, auc_scores, gmeans, thresholds]

# CALCOLA THRESHOLD DA ROC CURVE
def calcola_roc_curve_threshold(fpr, tpr, auc_score, gmeans, thresholds):
	max_gmean_index = np.argmax(gmeans)
	max_gmean = gmeans[max_gmean_index]
	max_gmean_thresh = thresholds[max_gmean_index]
	return [max_gmean_index, max_gmean, max_gmean_thresh]

# PLOT ROC-AUC CURVE
def plot_roc_auc_curve(fpr, tpr, auc_score, max_gmean_index, max_gmean, max_gmean_thresh, old_max_gmean_thresh=False):
	plt.figure(figsize=(2.5, 2.5))
	#plt.title("Receiver Operating Characteristic")
	plt.plot([0, 1], [0, 1],'r--', label="Bisector")
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
	#plt.scatter(fpr[max_gmean_index], tpr[max_gmean_index], marker='o', color='green', label='Best threshold = %0.2f' % max_gmean_thresh)
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel("TPR (Sensitivity)")
	plt.xlabel("FPR (1 - Specificity)")
	plt.legend(loc='best')
	plt.grid()


# PLOT CONFUSION MATRIX
def plot_confusion_matrix(confusion):
	# Absolute
	fig, ax = plt.subplots(figsize=(2.5, 5))
	ax.matshow(confusion, cmap=plt.cm.Blues)
	for i in range(confusion.shape[0]):
		for j in range(confusion.shape[1]):
			ax.text(x=j, y=i, s=confusion[i, j], va='center', ha='center', size='medium', color='white' if i==j==0 else 'black')
	alpha = ['False', 'True']
	ax.set_xticklabels(['']+alpha)
	ax.set_yticklabels(['']+alpha)
	plt.xlabel('Predictions')
	plt.ylabel('Actuals')
	#plt.title('Confusion Matrix')
	plt.grid(False)
	plt.show()

	# Normalized
	fig, ax = plt.subplots(figsize=(2.5, 5))
	ax.matshow(confusion, cmap=plt.cm.Blues)
	for i in range(confusion.shape[0]):
		for j in range(confusion.shape[1]):
			ax.text(x=j, y=i, s=float("{:.2f}".format(confusion[i, j] / np.sum(confusion))), va='center', ha='center', size='medium', color='white' if i==j==0 else 'black')
	alpha = ['False', 'True']
	ax.set_xticklabels(['']+alpha)
	ax.set_yticklabels(['']+alpha)
	plt.xlabel('Predictions')
	plt.ylabel('Actuals')
	#plt.title('Confusion Matrix')
	plt.grid(False)
	plt.show()


# PLOT IMPORTANCES
def plot_importances(feature_importances, columns):
	feats = {}
	for feature, importance in zip(columns, feature_importances):
		feats[feature] = importance
	importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'}).sort_values(by='Gini-importance', ascending=False)
	# PLOT
	importances.plot(kind='bar', rot=90, figsize=(10,5), grid=True, title='Gini-importances (from fitted model)')


# COMPUTE SHAP EXPLAINER
def calcola_shap(model, X_test, y_test, tree=True, fit_and_save=True, model_name="", n_samples=-1, probability=True, plot=False):
	#shap.initjs()

	X_test_shap, y_test_shap = X_test.to_numpy(dtype=object), y_test.to_numpy(dtype=object)
	#if n_samples != -1:
	#	X_test_shap, y_test_shap = shap.sample(X_test_shap, n_samples), shap.sample(y_test_shap, n_samples)

	# EXPLAINER
	if fit_and_save:
		if tree:
			explainer = shap.TreeExplainer(model)#, nsamples=nsamples)
		else:
			#if n_samples != -1:
			#	explainer = shap.KernelExplainer(model.predict, X_test_shap)
			#else:
			if probability:
				explainer = shap.KernelExplainer(model.predict_proba, X_test_shap, nsamples=n_samples)
			else:
				explainer = shap.KernelExplainer(model.predict, X_test_shap, nsamples=n_samples)
		shap_values = explainer.shap_values(X_test_shap)
		if model_name != "":
			dump(shap_values, model_name + "/shap" + ".joblib", compress=9)
	else:
		shap_values = load(model_name + "/shap" + ".joblib")
	
	original_shape = np.array(shap_values).shape

	#shap_values_reshaped = np.array(shap_values).reshape(original_shape[1], original_shape[0], original_shape[-1])
	if len(original_shape) != 2:
		shap_values_transposed = np.array(shap_values).transpose(0, 1, 2)
	else:
		shap_values = [np.array(shap_values), np.array(shap_values)]
		shap_values_transposed = np.array(shap_values).transpose(0, 1, 2)
	shap_v = shap_values_transposed

	# PLOT MEAN IMPORTANCES
	#shap.summary_plot(list(shap_v[:,:,:]), features=X_test, class_names=np.unique(y_test_shap), plot_type='bar', max_display=X_test.shape[1], plot_size=[4,18])
	#if tree:
	#	shap.summary_plot(shap_v[0], features=X_test, plot_type='dot', max_display=X_test.shape[1], plot_size=[12,15])
	#else:
	if plot:
		shap.summary_plot(shap_v[1], features=X_test, plot_type='dot', max_display=X_test.shape[1], plot_size=[8,12])

	# PRINT IMPORTANCE PERCENTAGES
	feature_importances = pd.DataFrame(list(zip(X_test.columns, np.abs(shap_values[0]).mean(0))), columns=['Feature','Importance'])
	feature_importances.sort_values(by=['Importance'], ascending=False, inplace=True)
	sum_feature_importance = np.sum(feature_importances['Importance'])
	feature_importances['Importance'] = feature_importances['Importance'] / sum_feature_importance
	feature_importances.sort_values(by=['Importance'], ascending=False, inplace=True)
	return feature_importances
