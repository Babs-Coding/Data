# Scaling data
def normalize(df):
	import numpy as np
	# select all numeric columns of df except 'SALE PRICE'
  	allcols = df.columns.values.tolist()
	num_cols = []
  	for col in allcols:
		if(df[col].dtype in [np.int64, np.int32, np.float64]):
  			num_cols.append(col)
      # normalize the dataset using this transformation
	df_norm = (df[num_cols]-df[num_cols].min())/(df[num_cols].max()-df[num_cols].min())
    	return df_norm
#GROSS_SQUARE_FEET+LAND_SQUARE_FEET+TOTAL_UNITS+RESIDENTIAL_UNITS
# Indentifying ouliers
def id_outlier(df):
    ## Create a vector of 0 of length equal to the number of rows
    temp = [0] * df.shape[0]
    ## test each outlier condition and mark with a 1 as required
    for i, x in enumerate(df['SALE PRICE']):
        if (x < 60000): temp[i] = 1
    for i, x in enumerate(df['GROSS SQUARE FEET']):
        if (x > 80000): temp[i] = 1
    for i, x in enumerate(df['LAND SQUARE FEET']):
        if (x > 10000): temp[i] = 1
    for i, x in enumerate(df['TOTAL UNITS']):
        if (x < 10): temp[i] = 1      
    df['outlier'] = temp # append a column to the data frame
    return df


def clean_auto(pathName, fileName = "Automobile_price_data_Raw.csv"):
    ## Load the data
    import pandas as pd
    import numpy as np
    import os

    ## Read the .csv file
    pathName = pathName
    fileName = fileName
    filePath = os.path.join(pathName, fileName)
    df = pd.read_csv(filePath)

    ## Convert some columns to numeric values
    cols = ['price', 'bore', 'stroke',
          'horsepower', 'peak-rpm']
    df[cols] = df[cols].convert_objects(convert_numeric = True)

    ## Drop unneeded columns
    drop_list = ['symboling', 'normalized-losses']
    df.drop(drop_list, axis = 1, inplace = True)


    ## Remove duplicate rows
    df.drop_duplicates(inplace = True)

    ## Remove rows with missing values
    df.dropna(axis = 0, inplace = True)

    ## Compute the log of the auto price
    df['SALE PRICE'] = np.log(df.price)

    ## Create a column with new levels for the number of cylinders
    df['num-cylinders'] = ['four-or-less' if x in ['two', 'three', 'four'] else
                                 ('five-six' if x in ['five', 'six'] else
                                 'eight-twelve') for x in df['num-of-cylinders']]
    ## Removing outliers
    df = id_outlier(df)  # mark outliers
    df = df[df.outlier == 0] # filter for outliers
    df.drop('outlier', axis = 1, inplace = True)

    ###
    df.to_csv('E:/CO3093/Datasets/cleaned_autoprice_data.csv')
    return df
