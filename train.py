from rgb_regression_model import RGBRegressionModel

if __name__ == '__main__':
    RGBRegressionModel(
        train_image_path=r'./train',i
        validation_image_path=r'/validation',
        input_shape=(64, 64, 3),
        lr=0.01,
        epochs=500,
        batch_size=32).fit()
