import awkward as ak
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, 
                             roc_curve, auc, precision_recall_curve, f1_score,
                             precision_score, recall_score, classification_report)
import matplotlib.pyplot as plt
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import xgboost as xgb
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import argparse

# Command Line arguments ----> Usage: python3 <python_file.py> --choice <YOUR_CHOICE>
parser= argparse.ArgumentParser(description="HyperParameter choices")
parser.add_argument(
    "--choice",
    type=int,
    default=1,
    help="choice number"
)
parser.add_argument("--dataset", type=str, default="ttbar_SemiLeptonic",help="Put the dataset name")
args= parser.parse_args()
choices=args.choice
dataset_name = args.dataset

# Define Choices:
if choices == 1:
    my_choice = "choice1"
elif choices == 2:
    my_choice = "choice2"
else:
    my_choice = "choice3"

# Create output directory
os.makedirs(f'output/{my_choice}', exist_ok=True)
os.makedirs(f'output/{my_choice}/Feature_correlation', exist_ok=True)
os.makedirs(f'output/{my_choice}/Best_Parameters', exist_ok=True)
os.makedirs(f'output/{my_choice}/roc-auc', exist_ok=True)
os.makedirs(f'output/{my_choice}/bdt-score', exist_ok=True)
os.makedirs(f'output/{my_choice}/Feature_Importance', exist_ok=True)

print("="*50)
print("BDT CLASSIFICATION FOR QQ VS NONQQ")
print("="*50)

use_gpu= False

#=================================
#    LOAD DATASET
#==================================
t0 = time.time()
print("\n[1] Loading data.....")
filename = f"{dataset_name}.parquet"
data = ak.from_parquet(filename)
data_flat = data[f"{dataset_name}"]
df = ak.to_dataframe(data_flat)

print(f"Total events loaded: {len(df)}")
print(f"Total number of Columns: {len(df.columns)}")
print(df.head())

#=======================================
#    LABEL ENGINEERING AND CLASS
#======================================
print("\n[2] Creating Binary labels")
df["y_binary"] = df["sum_pdgId"].apply(lambda x:1 if x==0 else 0)

number_of_events = {
    "QQBAR": len(df[df["y_binary"] == 1]),
    "NON_QQBAR": len(df[df["y_binary"]==0])
}

print("=====================CLASS DISTRIBUTION=================================")
print(f"  QQ_BAR (Class 1): {number_of_events['QQBAR']}")
print(f"  NON_QQ_BAR (Class 0): {number_of_events['NON_QQBAR']}")
print(f"  Imbalance ratio: {max(number_of_events.values()) / min(number_of_events.values()):.2f}")
print("========================================================================")

#==================================
#      BALANCING SAMPLE
#==================================
print("\n[3] BALANCING SAMPLE")
print("Usage: Take equal number of events from each class")
print("We shall take the value from Class 1 which is our QQBAR.... ")

length_class_1 = number_of_events["QQBAR"]

class_sig = df[df["y_binary"] == 1]
class_bkg = df[df["y_binary"] == 0]

def balancing_sample(class_name, length_of_sample):
    return class_name.sample(n=length_of_sample, random_state=42, replace=False)

df_class0 = balancing_sample(class_bkg,length_class_1)
df_class1 = balancing_sample(class_sig,length_class_1)

df_balanced = pd.concat([df_class1, df_class0], ignore_index=True)

print("Size of the balanced dataset : ",len(df_balanced))
print("Number of Unique rows : ", len(df_balanced.drop_duplicates()))  # <------ A simple debug !!

#==============================================
#   TRAIN-TEST-SPLIT(70% Training 30 % Testing)     
#==============================================
print("\n[4] Splitting df_balanced data for training and testing")
df_train, df_test = train_test_split(
    df_balanced,      # splits this data into training and testing
    test_size=0.3,    # in this ratio 70 percent for training and 30 percent for testing
    random_state=42,  # Everytime you run a program it selects the same rows for splitting , without this everytime you run program-->different rows selected for train/test , you cannot reproduce results
    stratify=df_balanced["y_binary"] # this ensure that you have same percentage of class 0 and class 1 events both in your train and test dataframe samples
)

print("="*60)
print("UNIQUENESS OF TRAIN-TEST-SPLIT")
print("No. of Unique events in train :", len(df_train.drop_duplicates()))
print("No. of Unique events in test  : ", len(df_test.drop_duplicates()))
print(f"No. of events in the training set : {len(df_train)} events")
print(f"No. of events in the testing set :  {len(df_test)} events")


print(f"\nClass distribution in splits")
print(f" Train: {df_train['y_binary'].value_counts().to_dict()}")
print(f" Test : {df_test['y_binary'].value_counts().to_dict()}")

#===========================================
#   FEATURE ENGINEERING
#===========================================
print("\n[5] Feature Engineering")
#features = ["FW1","FW2","FW3","pT_Sum","nJet","planarity","alignment","Sxy","Syy","Sxz","Syy","Syz","Szz","p2in","p2out","Sphericity","AL"]

#features = ["FW1","Sxy","Syy","AL","p2in","planarity","pT_Sum","nJet"]
features = ["FW1","Sxz","Szz","AL","p2in","planarity","pT_Sum","nJet","delta_R","dphi_lb"]

print(f"Number of features : {len(features)}")
print(f"Features: {features}")

#Extract features and labels
X_train = df_train[features]
Y_train = df_train["y_binary"]

X_test = df_test[features]
Y_test = df_test["y_binary"]

print("\n[6] Performing Data quality Checks")

print(f"Nans in X_train :\n{np.isnan(X_train).sum()}")
print(f"Nans in X_test :\n{np.isnan(X_test).sum()}")

#=============================================
#  IMPUTATION ON MISSING VALUES
#=============================================
print("\n[7] Imputing missing values")

#Imputation
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print(f"After imputing, nans in X_train:\n{np.isnan(X_train).sum()}")
print(f"After imputing, nans in X_test:\n{np.isnan(X_test).sum()}")

#=============================================
# FEATURE-FEATURE CORRELATION
#=============================================
print("\n[8] Feature-Feature Correlation")
feature_df = pd.DataFrame(X_train, columns = features)
corr_features = feature_df.corr()
print(corr_features)

plt.figure(figsize=(12, 10))
plt.imshow(corr_features, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation Coefficient")
plt.title("Feature-Feature Correlation Matrix")
plt.yticks(range(len(features)), features)
plt.tight_layout()
plt.savefig(f"output/{my_choice}/Feature_correlation/feature_feature_correlation.png",dpi=300,bbox_inches="tight")
plt.close()


#=============================================
#   MODEL TRAINING
#=============================================
print("\n[9] Model training first with a single model")

use_tree_method = "gpu_hist" if use_gpu else "hist"
xgb_binary = xgb.XGBClassifier(
    tree_method = use_tree_method,
    random_state = 42
)

start_time= time.time()
xgb_binary.fit(X_train, Y_train)
end_time = time.time()

print(f"It took {end_time-start_time} to train a single model")

print("Your selected choice is :", choices)
#============================================
#      HYPERPARAMETER OPTIMIZATION
#============================================
print("\n[10] Hyperparameter Optimization with GridSearchCV")
print("  This may take several minutes...")

param_grid_1 = {
    "n_estimators":[50, 100, 200],
    "learning_rate":[0.01, 0.1, 0.2],
    "max_depth":[3,4,5],
    "min_child_weight":[1,2,4],
    "subsample":[0.8,0.9,1.0],
    "colsample_bytree":[0.8,0.9,1.0]

}


param_grid_2={
    "n_estimators": [250],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3,4,5],
    "min_child_weight": [1,3,5],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    #"gamma": [0, 0.1, 0.3],
    #"reg_lambda": [1, 2, 5],
    #"reg_alpha": [0, 0.1, 0.5]
}
'''
param_grid_2={
    "n_estimators": [250],
    "learning_rate": [0.1],
    "max_depth": [5],
    "min_child_weight": [3],
    "subsample": [1.0],
    "colsample_bytree": [0.7],
    #"gamma": [0, 0.1, 0.3],
    #"reg_lambda": [1, 2, 5],
    #"reg_alpha": [0, 0.1, 0.5]
}
'''
param_grid_3 = {
    "n_estimators": [300,350],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3,4,5],
    "min_child_weight": [3, 5],
    "subsample": [0.7, 0.85],
    "colsample_bytree": [0.7, 0.85],
    #"gamma":[0,0.1], # added new
    #"reg_alpha":[0,0.1] #added new 
}

param_grid_dict = {
    1: param_grid_1,
    2: param_grid_2,
    3: param_grid_3
}

param_grid = param_grid_dict[choices]

grid_search = GridSearchCV(
    estimator = xgb_binary,
    param_grid=param_grid,
    cv =3,
    scoring="accuracy", # was accuracy before
    n_jobs=-1,
    #verbose=2,
    #return_train_score=True
)

start_time_grid_search = time.time()
grid_search.fit(X_train,Y_train)
optimization_time = time.time() - start_time_grid_search

# Print the best parameters with their value:
print("    Best Parameters with their values")
for param, value in grid_search.best_params_.items():
    print(f"    {param}: {value}")

#store the best parameters to a txt file 
best_params_txt_output = f"output/{my_choice}/Best_Parameters/best_params.txt"
with open(best_params_txt_output, "w") as f:
    f.write("Best Parameters with their values\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"{param}: {value}\n")


best_params = grid_search.best_params_
#Get the best model
print("\n[11] Training the best model")
xgb_binary_best = xgb.XGBClassifier(**best_params, tree_method=use_tree_method,random_state=42)
xgb_binary_best.fit(X_train, Y_train)

#====================================
#    MAKING PREDICTIONS
#====================================
print("\n[12] Making Predictions")

#Predictions on the training set
Y_train_pred = xgb_binary_best.predict(X_train)
Y_train_pred_proba = xgb_binary_best.predict_proba(X_train)[:,1]


#Predictions on the test set
Y_test_pred = xgb_binary_best.predict(X_test)
Y_test_pred_proba = xgb_binary_best.predict_proba(X_test)[:,1]

# DEBUG 
probs_col_1 = xgb_binary_best.predict_proba(X_test)[:,1]
background_scores = probs_col_1[Y_test == 0]
signal_scores = probs_col_1[Y_test == 1]

print(f"Average score for Background events: {background_scores.mean():.4f}")
print(f"Average score for Signal events:     {signal_scores.mean():.4f}")
#=====================================
# PERFORMANCE METRICS
#=====================================
print("\n[13] Performance Metrics")

def calculate_metrics(y_true,y_pred,y_pred_proba,set_name):
    accuracy = accuracy_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true,y_pred_proba)
    
    print(f"\n{set_name} Set Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

train_metrics = calculate_metrics(Y_train, Y_train_pred, Y_train_pred_proba, "Training")
test_metrics = calculate_metrics(Y_test, Y_test_pred ,Y_test_pred_proba, "Testing")

print("TEST_METRICS RESULTS : ")
print(f"accuracy:{test_metrics['accuracy']:.2f}")
print(f"roc_auc score: {test_metrics['roc_auc']:.2f}")

print("\n[15.a] Overfitting Check")
print(f"    Train-Test ROC-AUC difference: {abs(train_metrics['roc_auc'] - test_metrics['roc_auc']):.4f}")
print(f"    Train-Test Accuracy difference: {abs(train_metrics['accuracy'] - test_metrics['accuracy']):.4f}")

#==================================================
#    ROC CURVES
#==================================================
print("\n[14] ROC Curves")

def plot_roc_curves():
    #Training roc curve 
    fpr_train, tpr_train, threshold_train = roc_curve(Y_train, Y_train_pred_proba)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    #Test ROC
    fpr_test, tpr_test, threshold_test = roc_curve(Y_test, Y_test_pred_proba)
    roc_auc_test = auc(fpr_test, tpr_test)
    
    #Plot only for test
    plt.figure(figsize=(10,8))
    plt.plot(fpr_test,tpr_test,color="darkorange",lw=2,label=f"ROC Curve (area: {roc_auc_test:.3f})")
    plt.plot([0,1],[0,1],color="navy",lw=2,linestyle="--")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Reciever Operating Characteristics(ROC)")
    plt.legend(loc="lower right")
    #plt.savefig(f"output/plots/roc_curve_test_only_binary.png")
    plt.savefig(f"output/{my_choice}/roc-auc/roc_curve_test_only_binary_for_{dataset_name}.png") 
    plt.close()
    
    
    #Compare test ROC vs train ROC 
    plt.figure(figsize=(12,9))
    plt.plot(fpr_train,tpr_train,color="blue",lw=3,linestyle="--",label=f"Training (AUC = {roc_auc_train:.3f})")
    plt.plot(fpr_test,tpr_test,color="red",lw=3,linestyle="-",label=f"Testing (AUC = {roc_auc_test:.3f})")
    plt.plot([0,1],[0,1],color="gray",lw=2.5,linestyle="--",label="Random Classifier")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate",fontsize=22,fontweight="bold")
    plt.ylabel("True Positive Rate",fontsize=22,fontweight="bold")
    plt.title("Training vs Testing ROC Curves",fontsize=24,fontweight="bold",pad=20)
    plt.legend(loc="lower right",fontsize=22,frameon=True,fancybox=True,shadow=True,framealpha=0.95)
    plt.tick_params(axis="both",which="major",labelsize=22,width=1.6,length=7,direction="in")
    plt.grid(True, alpha=0.3,linestyle="--",linewidth=0.8)
    plt.tight_layout()
    #plt.savefig("output/plots/roc_curve_training_vs_testing.png", dpi=300,bbox_inches="tight")
    plt.savefig(f"output/{my_choice}/roc-auc/roc_curve_training_vs_testing_{features[-1]}_for_{dataset_name}.png",dpi=1000,bbox_inches="tight")
    plt.close()
    
    return fpr_train,tpr_train,fpr_test,tpr_test,threshold_train, threshold_test

fpr_train, tpr_train ,fpr_test, tpr_test, threshold_train, threshold_test = plot_roc_curves()
print("ROC CURVES SAVED IN THE DESIRED LOCATION")

#=============================================================
#   BDT Score Distribution
#=============================================================
print("\n[15] BDT score distribution")
plt.figure(figsize=(12,6))

bins_impr = np.linspace(0,1,51)
# Training set
plt.subplot(2, 1, 1)
plt.hist(Y_train_pred_proba[Y_train == 1], bins=bins_impr, alpha=0.5, 
         label='QQ-bar (Class 1)', color='blue', density=True)
plt.hist(Y_train_pred_proba[Y_train == 0], bins=bins_impr, alpha=0.5, 
         label='Non-QQ-bar (Class 0)', color='red', density=True)
plt.xlabel('BDT Score')
plt.ylabel('Normalized Events')
plt.title('BDT Score Distribution - Training Set')
plt.legend()
plt.grid(True, alpha=0.3)

# Test set
plt.subplot(2, 1, 2)
plt.hist(Y_test_pred_proba[Y_test == 1], bins=bins_impr, alpha=0.5, 
         label='QQ-bar (Class 1)', color='blue', density=True)
plt.hist(Y_test_pred_proba[Y_test == 0], bins=bins_impr, alpha=0.5, 
         label='Non-QQ-bar (Class 0)', color='red', density=True)
plt.xlabel('BDT Score')
plt.ylabel('Normalized Events')
plt.title('BDT Score Distribution - Test Set')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig("output/plots/BDT_score_distribution_for_train_and_test.png")
plt.savefig(f"output/{my_choice}/bdt-score/BDT_score_distribution_for_train_and_test_{features[-1]}_for_{dataset_name}.png")

print("\n[15a] BDT score distribution (Train vs Test)")
plt.figure(figsize=(12,9))

min_score = min(Y_train_pred_proba.min(),Y_test_pred_proba.min())
max_score = max(Y_train_pred_proba.max(),Y_test_pred_proba.max())

bins = np.linspace(min_score, max_score,51)

print(f"For training min score is  : {Y_train_pred_proba.min()}")
print(f"For training max score is  : {Y_train_pred_proba.max()}")

print(f"For testing min score is  : {Y_test_pred_proba.min()}")
print(f"For testing max score is  : {Y_test_pred_proba.max()}")


# QQ-bar (Class 1)
plt.hist(Y_train_pred_proba[Y_train == 1], bins=bins, density=True,
         histtype='step', color='blue', linestyle='--', linewidth=3.5,
         label='QQ-bar (Train)',alpha=0.9)
plt.hist(Y_test_pred_proba[Y_test == 1], bins=bins, density=True,
         histtype='step', color='blue', linestyle='-', linewidth=3.5,
         label='QQ-bar (Test)',alpha=0.9)

# Non-QQ-bar (Class 0)
plt.hist(Y_train_pred_proba[Y_train == 0], bins=bins, density=True,
         histtype='step', color='red', linestyle='--', linewidth=3.5,
         label='Non-QQ-bar (Train)')
plt.hist(Y_test_pred_proba[Y_test == 0], bins=bins, density=True,
         histtype='step', color='red', linestyle='-', linewidth=3.5,
         label='Non-QQ-bar (Test)',alpha=0.9)

plt.xlabel('BDT Score', fontsize=22, fontweight ="bold")
plt.ylabel('Normalized Events',fontsize=22,fontweight="bold")
plt.title('BDT Score Distribution: Train vs Test')
plt.legend(loc='upper left', ncol=1, fontsize=22, frameon=True, fancybox=True, shadow=True, framealpha=0.95)
plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
plt.tick_params(axis="both",which="major", labelsize=22,width=1.5,length=7,direction="in")

plt.tight_layout()
plt.savefig(f"output/{my_choice}/bdt-score/BDT_score_distribution_train_vs_test_{features[-1]}_for_{dataset_name}.png")
plt.show()

#Save the BDT scores of the training and testing in a csv file
os.makedirs(f"output/{my_choice}/bdt-score/Train_Test_bdt_scores", exist_ok =True)
train_score_df = pd.DataFrame({
    "BDT_Score":Y_train_pred_proba,
    "True_Label":Y_train    
})

test_score_df = pd.DataFrame({
    "BDT_Score":Y_test_pred_proba,
    "True_Label":Y_test
})


train_csv_path = f"output/{my_choice}/bdt-score/Train_Test_bdt_scores/2016preVFP_training_scores.csv"
test_csv_path = f"output/{my_choice}/bdt-score/Train_Test_bdt_scores/2016preVFP_testing_scores.csv"

train_score_df.to_csv(train_csv_path, index=False)
test_score_df.to_csv(test_csv_path, index=False)

print(f"\n[15b] Saved BDT scores for {dataset_name}:")
print(f" → Train scores: {train_csv_path}")
print(f" → Test scores : {test_csv_path}")


#======================================================
# FEATURE IMPORTANCE
#======================================================
print("\n[16] Final Feature Importance Analysis")
feature_importance_final = pd.DataFrame({
    "feature":features,
    "importance":xgb_binary_best.feature_importances_
}).sort_values("importance",ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance_final.head(13))

# Save to CSV
feature_importance_final.to_csv(f'output/{my_choice}/Feature_Importance/feature_importance_for_{dataset_name}.csv', index=False)

plt.figure(figsize=(12, 10))
top_n = min(20, len(features))
plt.barh(range(top_n), feature_importance_final['importance'][:top_n].values)
plt.yticks(range(top_n), feature_importance_final['feature'][:top_n])
plt.xlabel('Feature Importance ')
plt.title(f'Top {top_n} Feature Importances - Final Model')
plt.tight_layout()
plt.savefig(f'output/{my_choice}/Feature_Importance/final_feature_importance_for_{dataset_name}.png', dpi=300, bbox_inches='tight')
plt.close()

#Save your best model for later use 
#joblib.dump(xgb_binary_best, f"Final_best_bdt_model_temp_trained_for_{my_choice}.pkl")

'''
joblib.dump({
    "model":xgb_binary_best,
    "features": features,
    "imputer":imputer
},f"Final_best_bdt_model_temp_trained_for_{my_choice}_for_{features[-1]}.pkl")
'''
model_path = "trained_model_for_ttbar.xgb"
xgb_binary_best.get_booster().save_model(model_path)

joblib.dump({"features":features},"trained_features.pkl")

print("Your model was saved")
