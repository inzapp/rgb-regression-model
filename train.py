from rgb_regression_model import RGBRegressionModel

if __name__ == '__main__':
    RGBRegressionModel(
        train_image_path=r'C:\inz\tmp\7_600_jhpark',
        validation_split=0.2,
        input_shape=(64, 64, 3),
        lr=0.01,
        epochs=300,
        batch_size=32).fit()
