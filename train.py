from rgb_regression_model import RGBRegressionModel

if __name__ == '__main__':
    RGBRegressionModel(
        train_image_path=r'C:\inz\train_data\car_color_regression\train',
        validation_image_path=r'C:\inz\train_data\car_color_regression\validation',
        input_shape=(128, 128, 3),
        train_type='one_color',
        lr=0.001,
        momentum=0.9,
        batch_size=32,
        iterations=300000,
        training_view=True).fit()
