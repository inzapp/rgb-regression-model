from rgb_regression_model import RGBRegressionModel

if __name__ == '__main__':
    RGBRegressionModel(
        # pretrained_model_path=r'checkpoints/model_100000_iter_0.0491_val_loss.h5',
        train_image_path=r'C:\inz\train_data\all_color\train\car',
        validation_image_path=r'C:\inz\train_data\all_color\validation\car_color',
        input_shape=(128, 128, 3),
        train_type='one_color',
        lr=0.001,
        warm_up=0.5,
        momentum=0.9,
        batch_size=32,
        iterations=200000,
        training_view=True).fit()
        # training_view=True).predict_validation_images()
