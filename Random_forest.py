# Split the data into features (X) and labels (y)
X_train = df_train[['twi','spi','slope','riverslope25050','trslope25050','profile','plan','lulc','ndvi','geology','fault700100','aspect','dem','wgt','gs']]
X_train = X_train.values
y_train = df_train['Hazard'].values

# Load the test data into a pandas dataframe
df_test = pd.read_csv('datasets/mcc_vy/testing_set_4.csv')

# Split the data into features (X) and labels (y)
X_test = df_test[['twi','spi','slope','riverslope25050','trslope25050','profile','plan','lulc','ndvi','geology','fault700100','aspect','dem','wgt','gs']]
X_test = X_test.values
y_test = df_test['Hazard'].values
# create a RandomForestRegressor model
model_rr = RandomForestRegressor(n_estimators=1000, random_state=123)
model_rr.fit(X_train, y_train)
# Search for the best threshold and give the best roc-auc score
def get_best_threshold(y_probs, y_true):
    # Define a range of thresholds
    thresholds = np.arange(0, 1.05, 0.05)

    # Find the best threshold based on the accuracy score
    best_threshold = 0
    best_accuracy = 0
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        accuracy = roc_auc_score(y_true.flatten(), y_pred.flatten())
        if accuracy > best_accuracy:
            best_threshold = threshold
            best_accuracy = accuracy

    return round(best_threshold, 2)
# Create an empty DataFrame to store the pixel values
df = pd.DataFrame()

# Loop through the TIFF files and read them into the DataFrame
# List the folder
parent_dir = 'D:/Projects/Supervised ML/lsm/datasets/factors/'
tif_files = os.listdir('datasets/factors/')
for i in range(len(tif_files)):
    file_path = parent_dir+tif_files[i]
    tif_data = tifffile.imread(file_path)
    tif_data = tif_data.flatten() # Flatten the 2D array into a 1D array
    column_name = file_path.split('/')[-1].split('_')[-1].split('.')[0]
    df[f'{column_name}'] = tif_data # Add the 1D array as a column in the DataFrame
# Now you can make predictions on the DataFrame
data = df.values
preds = model.predict(data)
preds = (preds-preds.min())/(preds.max()-preds.min())
# You can also transform the DataFrame back into a TIFF file
# Let's assume that the original TIFF files had the same dimensions
tif_get_shape = 'datasets/factors/[CLASSIFIED]_MCC_VY_dem.tif'
# Open the factor file to get its CRS and other metadata
with rasterio.open(tif_get_shape) as factor_file:
    factor_crs = factor_file.crs
    factor_transform = factor_file.transform
    factor_width = factor_file.width
    factor_height = factor_file.height
    factor_dtype = factor_file.dtypes[0] # Assume all bands have the same dtype
# Create a new TIFF file with the same CRS as the factor file
with rasterio.open('output/lsm7_enre702.tif', 'w', 
                   driver='GTiff', 
                   height=factor_height, 
                   width=factor_width, 
                   count=1, 
                   dtype=rasterio.float32,
                   nodata=nodata_value,
                   crs=factor_crs, 
                   transform=factor_transform) as new_tif_file:
    new_tif_data = preds.reshape(factor_file.shape)
    new_tif_file.write(new_tif_data, 1) # Write the data to band 1

