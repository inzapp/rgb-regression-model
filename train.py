from rgb_regression_model import RGBRegressionModel

if __name__ == '__main__':
    RGBRegressionModel(
        train_image_path=r'C:\inz\train_data\car_rgb_regression_one_color\train',
        validation_image_path=r'C:\inz\train_data\car_rgb_regression_one_color\validation',
        input_shape=(64, 64, 3),
        lr=0.001,
        momentum=0.9,
        decay=5e-5,
        burn_in=1000,
        batch_size=32,
        iterations=300000,
        training_view=False).fit()
