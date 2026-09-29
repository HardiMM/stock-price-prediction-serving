from src.data_loader import download_stock_data
from src.preprocess import create_features
from src.train import train_model


def run_pipeline():

    print("=" * 50)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 50)

    download_stock_data()

    print("\n" + "=" * 50)
    print("STEP 2: PREPROCESSING DATA")
    print("=" * 50)

    create_features()

    print("\n" + "=" * 50)
    print("STEP 3: TRAINING MODEL")
    print("=" * 50)

    train_model()

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()