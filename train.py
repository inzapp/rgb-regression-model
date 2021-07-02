from rgb_regression_model import RGBRegressionModel

if __name__ == '__main__':
    RGBRegressionModel(
        train_image_path=r'./train',
        validation_image_path=r'./validation',
        input_shape=(32, 32, 3),
        lr=0.01,
        epochs=300,
        batch_size=32).fit()
