"""Пример: python -m ml.train_url data/raw/url_dataset.csv --url-column URL --label-column label"""
import argparse
from ml.url_model import train_url_model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()
    print(train_url_model(args.csv, args.url_column, args.label_column))
if __name__ == "__main__": main()
