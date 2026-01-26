import pandas as pd
from scipy.io import arff

def arff_to_csv(arff_path, csv_path):
    data, meta = arff.loadarff(arff_path)
    df = pd.DataFrame(data)

    # Decode byte strings (very common in UCI datasets)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode("utf-8")

    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")


if __name__ == '__main__':

    uci = "./Datasets/phishing+websites/TrainingDataset.arff"
    savedToCSV = "./Datasets/uci.csv"

    arff_to_csv(uci, savedToCSV)

    
