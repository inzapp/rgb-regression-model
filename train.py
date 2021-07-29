from rgb_regression_model import RGBRegressionModel

if __name__ == '__main__':
    RGBRegressionModel(
        train_image_path=r'./train',
        validation_image_path=r'./validation',
        input_shape=(64, 64, 3),
        lr=0.001,
        burn_in=1000,
        batch_size=32,
        iterations=100000).fit()
